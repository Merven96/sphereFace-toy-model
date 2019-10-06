import os
import scipy.io as io
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from feature_dataloader import FeatureDataset
from net_sphere_toy import transform_net
from net_sphere_toy import conv_transform_net

def parameter():
    parser = argparse.ArgumentParser(description="Transformer")

    parser.add_argument('--train_list', nargs='+') # 4, 5, 6
    # parser.add_argument('--train_data_mat', type=str, default="./mat_file/toy_model/train_without_test/toy_feature_train.mat")
    parser.add_argument('--label_data_mat', type=str, default="./mat_file/toy_model/train_without_test/toy_feature_train_lr.mat")
    parser.add_argument('--test_label_mat', type=str, default=None)

    parser.add_argument('--use_gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--mid_num', type=int, default=4)
    parser.add_argument('--mid_dimen', type=int, default=4)

    parser.add_argument('--whether_testset', type=bool, default=False)
    # parser.add_argument('--test_input_mat', type=str, default=None)
    parser.add_argument('--whether_save', type=bool, default=False)
    parser.add_argument('--saving_title', type=str, default="")
    
    parser.add_argument('--type', type=str, default='fc')
    parser.add_argument('--seq_length', type=int, default=0)
    return parser


def train(train_data_mat_list, label_data_mat, net, batch_size, epoch_num, lr=0.001, test_input=None, test_label=None):
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        net = net.cuda()
        # optimizer = optimizer.cuda()

    net.train()
    train_loss = 0

    dataset_num = len(train_data_mat_list)
    data_num = np.shape(train_data_mat_list[0])[0]
    batch_num = int(dataset_num * data_num / batch_size)

    if test_input is not None and test_label is not None:
        test_input = torch.autograd.Variable(torch.from_numpy(test_input))
        test_label = torch.autograd.Variable(torch.from_numpy(test_label))
        

    for epoch in range(epoch_num):
        train_data_list_shuffle, label_data_shuffle = combine_and_shuffle_diff_set(train_data_mat_list, label_data_mat)

        if epoch >0 and epoch % 5 == 0:
            lr = 0.1 * lr
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for i in range(batch_num):
            # batch_end = i*batch_size + batch_size

            train_batch, label_batch = get_batch_from_different_set(train_data_list_shuffle, label_data_shuffle, batch_size)
            # print("test: ", np.shape(train_batch), np.shape(label_batch))

            train_batch = torch.autograd.Variable(torch.from_numpy(train_batch))
            label_batch = torch.autograd.Variable(torch.from_numpy(label_batch))
           

            # if batch_end < train_data.shape[0]:
            #     train_batch = torch.autograd.Variable(torch.from_numpy(train_data[i*batch_size:i*batch_size + batch_size, :]))
            #     label_batch = torch.autograd.Variable(torch.from_numpy(label_data[i*batch_size:i*batch_size + batch_size, :]))
            # else:
            #     train_batch = torch.autograd.Variable(torch.from_numpy(train_data[i*batch_size:, :]))
            #     label_batch = torch.autograd.Variable(torch.from_numpy(label_data[i*batch_size:, :]))


            if cuda:
                train_batch = train_batch.cuda()
                label_batch = label_batch.cuda()

            output_batch = net(train_batch)
            train_loss = loss_fn(label_batch.float(),\
                                 output_batch.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if(epoch==0 and i==0):
                print("start loss: {}".format(train_loss.item()))
                if test_input is not None and test_label is not None:
                    if cuda:
                        test_input = test_input.cuda()
                        test_label = test_label.cuda()
                    test_output = net(test_input)
                    test_loss = loss_fn(test_output.float(), \
                                        test_label.float())        
                    print("start test-loss: {}".format(test_loss.item()))
            else:
                print("epoch-{}  {}/{} loss: {} ".format(epoch, i, batch_num, train_loss.item()))

        if test_input is not None and test_label is not None:
                test_output = net(test_input)
                test_loss = loss_fn(test_output.float(), \
                                    test_label.float())        
                print("{}-epoch / {} train_loss: {}  ; test-loss: {}".\
                        format(epoch, epoch_num, train_loss.item(),\
                               test_loss.item()))
        else:
            print("{}-epoch / {} train_loss: {}".format\
                        (epoch, epoch_num, train_loss.item()))
            
    return net


def combine_and_shuffle(train_data_mat, label_data_mat):
    if not (np.shape(train_data_mat)[0] == np.shape(label_data_mat)[0]):
        print("error")
        return

    permutation = np.random.permutation(train_data_mat.shape[0])
    train_data_mat = train_data_mat[permutation, :]
    label_data_mat = label_data_mat[permutation, :]
    return train_data_mat, label_data_mat


def combine_and_shuffle_diff_set(input_mat_list, label_mat):
    dataset_num = len(input_mat_list)
    for i in range(dataset_num):
        if not (np.shape(input_mat_list[i])[0] == \
                np.shape(label_mat)[0]):
            print(np.shape(input_mat_list[i]), np.shape(label_mat))
            print("error")
            return
    permutation = np.random.permutation(input_mat_list[0].shape[0])
    label_mat = label_mat[permutation, :]
    for i in range(dataset_num):
        input_mat_list[i] = input_mat_list[i][permutation, :]
    return input_mat_list, label_mat



def get_batch_from_different_set(train_mat_list, label_mat, batch_size):
    dataset_num = len(train_mat_list)
    if not(batch_size % dataset_num == 0):
        print("ERROR: the batch_size is not suitable")
        return None

    # 确定从每个dataset-mat中取多少条向量
    num_per_set = int(batch_size / dataset_num)
    data_left = train_mat_list[0].shape[0]
    if data_left < num_per_set:
        num_per_set = dataleft

    # print("test:", np.shape(train_mat_list), type(num_per_set), num_per_set)
    train_batch = train_mat_list[0][0:num_per_set, :]
    train_mat_list[0] = np.delete(train_mat_list[0], range(num_per_set), axis=0)
    label_batch = label_mat[0:num_per_set, :]
    for i in range(dataset_num-1):
        train_batch = np.vstack((train_batch, \
                                 train_mat_list[i+1][0:num_per_set, :]))
        train_mat_list[i+1] = np.delete(train_mat_list[i+1], range(num_per_set), axis=0)
        label_batch = np.vstack((label_batch, label_mat[0:num_per_set, :]))

    label_mat = np.delete(label_mat, range(num_per_set), axis=0)

    return train_batch, label_batch
        

def save_model(model, filename):
    state = model.state_dict()
    for key in state: 
        state[key] = state[key].clone().cpu()
        torch.save(state, filename)


if __name__ == "__main__":
    args = parameter()
    args = args.parse_args()

    # for elem in args.train_input_list:
    #     print(type(elem), elem)

    dataset_num = len(args.train_list)
    file_prefix = "mat_file/high_dimension_model/resize"
    train_input_list = []
    test_input_list = []
    for elem in args.train_list:
        train_input_list.append(file_prefix + elem + '/high_feature_train_input_resize' + elem + '.mat')
        test_input_list.append(file_prefix + elem + '/high_feature_test_input_resize' + elem + '.mat')

    label_data = io.loadmat(args.label_data_mat)
    label_data = label_data['name']
    test_label = io.loadmat(args.test_label_mat)
    test_label = test_label['name']

    print("file_list: ")
    print("training-set input file list:")
    print(train_input_list)

    print("training-set label file list:")
    print(args.label_data_mat)

    print("testing-set input file list:")
    print(test_input_list)
    
    print("testing-set label file list:")
    print(args.test_label_mat)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    train_data = []
    for elem in train_input_list:
        train_data_elem = io.loadmat(elem)['name']
        print("training-data:", np.shape(train_data_elem))
        train_data.append(train_data_elem)

    test_input = []
    for elem in test_input_list:
        test_elem = io.loadmat(elem)['name']
        print("testing-data:", np.shape(test_elem))
        test_input.append(test_elem)


    # if(args.test_input_mat is not None):
    #     test_input = io.loadmat(args.test_input_mat)
    #     test_input = test_input['name']
    #     test_label = io.loadmat(args.test_label_mat)
    #     test_label = test_label['name']
    #     print("testing-data:", np.shape(test_input), np.shape(test_label))
    # else:
    #     test_input = None
    #     test_label = None

    feature_dimension = train_data[0].shape[1]


    if args.type == "conv":
        train_data = np.expand_dims(train_data, axis=1)
        label_data = np.expand_dims(label_data, axis=1)
        net = conv_transform_net(f_dimension=feature_dimension, \
                   mid_dimension=args.mid_dimen, mid_num=args.mid_num, \
                   seq_length=args.seq_length)
        if(args.test_input_mat is not None):
            test_input = np.expand_dims(test_input, axis=1)
            test_label = np.expand_dims(test_label, axis=1)
    else:
        net = transform_net(f_dimension=feature_dimension, \
                   mid_dimension=args.mid_dimen, mid_num=args.mid_num)

    # net = train(train_data, label_data, net, args.batch_size, args.epoch_num, args.learning_rate, test_input, test_label)
    net = train(train_data, label_data, net, args.batch_size, args.epoch_num, args.learning_rate)

    if(args.whether_save is True):
        save_name = args.saving_title+'transform_{}-midNum_{}-midDimen-epoch_{}.pth'.format(args.mid_num, args.mid_dimen, args.epoch_num)
        save_model(net, save_name)
