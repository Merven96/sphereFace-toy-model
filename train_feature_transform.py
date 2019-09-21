import os
import scipy.io as io
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from feature_dataloader import FeatureDataset
from net_sphere_toy import transform_net

def parameter():
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument('--train_data_mat', type=str, default="./mat_file/train_without_test/toy_feature_train.mat")
    parser.add_argument('--label_data_mat', type=str, default="./mat_file/train_without_test/toy_feature_train_lr.mat")
    parser.add_argument('--use_gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--mid_num', type=int, default=4)
    parser.add_argument('--mid_dimen', type=int, default=4)

    parser.add_argument('--test_input_mat', type=str, default=None)
    parser.add_argument('--test_label_mat', type=str, default=None)
    return parser

def train(train_data_mat, label_data_mat, net, batch_size, epoch_num, lr=0.001, test_input=None, test_label=None):
    dataset = FeatureDataset(train_data_mat, label_data_mat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    net.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_loss = 0

    data_num = np.shape(train_data_mat)[0]
    batch_num = int(data_num / batch_size)

    if test_input is not None and test_label is not None:
        test_input = torch.autograd.Variable(torch.from_numpy(test_input))
        test_label = torch.autograd.Variable(torch.from_numpy(test_label))
        

    for epoch in range(epoch_num):
        train_data, label_data = combine_and_shuffle(train_data_mat, label_data_mat)

        if epoch >0 and epoch % 5 == 0:
            lr = 0.1 * lr
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for i in range(batch_num):
            batch_end = i*batch_size + batch_size
            if batch_end < train_data.shape[0]:
                train_batch = torch.autograd.Variable(torch.from_numpy(train_data[i*batch_size:i*batch_size + batch_size, :]))
                label_batch = torch.autograd.Variable(torch.from_numpy(label_data[i*batch_size:i*batch_size + batch_size, :]))
            else:
                train_batch = torch.autograd.Variable(torch.from_numpy(train_data[i*batch_size:, :]))
                label_batch = torch.autograd.Variable(torch.from_numpy(label_data[i*batch_size:, :]))

            output_batch = net(train_batch)
            train_loss = loss_fn(label_batch.float(),\
                                 output_batch.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if(epoch==0 and i==0):
                print("start loss: {}".format(train_loss.item()))
                if test_input is not None and test_label is not None:
                    test_output = net(test_input)
                    test_loss = loss_fn(test_output.float(), \
                                        test_label.float())        
                    print("start test-loss: {}".format(test_loss.item()))

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

def save_model(model, filename):
    state = model.state_dict()
    for key in state: 
        state[key] = state[key].clone().cpu()
        torch.save(state, filename)

if __name__ == "__main__":
    args = parameter()
    args = args.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    train_data = io.loadmat(args.train_data_mat)
    train_data = train_data['name']
    label_data = io.loadmat(args.label_data_mat)
    label_data = label_data['name']
    print("training-data:", np.shape(train_data), np.shape(label_data))

    if(args.test_input_mat is not None):
        test_input = io.loadmat(args.test_input_mat)
        test_input = test_input['name']
        test_label = io.loadmat(args.test_label_mat)
        test_label = test_label['name']
        print("testing-data:", np.shape(test_input), np.shape(test_label))

    net = transform_net(f_dimension=2, \
            mid_dimension=args.mid_dimen, mid_num=args.mid_num)
    net = train(train_data, label_data, net, args.batch_size, args.epoch_num, args.learning_rate, test_input, test_label)

    save_model(net, 'transform_{}-midNum_{}-midDimen_v2.pth'\
                    .format(args.mid_num, args.mid_dimen))

