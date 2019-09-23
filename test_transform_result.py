from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from matlab_cp2tform import get_similarity_transform_for_cv2
from net_sphere_toy import transform_net
import net_sphere_toy

import time
import sys
from datasets import ImageDataset_two_dir_with_label

parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--model_file', type=str, default='./toy_model/sphere_20a_100_toy.pth')
parser.add_argument('--dataset_name_hr', type=str, default="../../../data/CASIA_10_train.zip", help='zip-file of hr dataset')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--landmark', type=str, default="./data/casia_landmark.txt")
parser.add_argument("--use_gpu", type=str, default='1')
parser.add_argument('--mid_num', type=int, default=4)
parser.add_argument('--mid_dimen', type=int, default=4)


args = parser.parse_args()
use_cuda = torch.cuda.is_available()


def alignment(src_img,src_pts):
    of = 2
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img



def read_img(filename,zfile, whether_align=True):
    position = filename.rfind('.zip:')
    zipfilename = filename[0:position+4]
    nameinzip = filename[position+5:]
    split = nameinzip.split('\t')
    nameinzip = split[0]
    classid = int(split[1])
    src_pts = []
    for i in range(5):
        src_pts.append([int(split[2*i+2]),int(split[2*i+3])])
    data = np.frombuffer(zfile.read(nameinzip),np.uint8)
    img = cv2.imdecode(data,1)

    return (img, classid)

def img_process(filename, img):
    """
    img是cv2读取好的图片，从filename中获取剪裁信息
    """
    position = filename.rfind('.zip:')
    zipfilename = filename[0:position+4]
    nameinzip = filename[position+5:]
    split = nameinzip.split('\t')
    nameinzip = split[0]
    classid = int(split[1])
    src_pts = []
    for i in range(5):
        src_pts.append([int(split[2*i+2]),int(split[2*i+3])])

    img = alignment(img,src_pts)
    img = img[2:2+112,2:2+96,:]

    img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    img = ( img - 127.5 ) / 128.0   
    return img


def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_noise(file_path, zfile, prob):
    img_cv, label = read_img(file_path, zfile)
    img_noise = img_cv.copy()

    img_noise = sp_noise(img_noise, prob)

    img_noise_np = img_process(file_path, img_noise)
    tensor_i = torch.from_numpy(img_noise_np).type(torch.FloatTensor).cuda()
    feature_down = net(tensor_i)
    feature_down = feature_down.cpu()
    feature_down = feature_down.detach().numpy()[0]
    return (feature_down, img_cv, img_noise, label)


def down_sample(file_path, zfile, downsample_factor, net):
    # 下面对file_path对应的图进行处理，获取新的特征向量（2维）
    
    img_cv, label = read_img(file_path, zfile)
    size1, size2, _ = np.shape(img_cv)
    img_downsample = img_cv.copy()

    img_downsample = cv2.resize(img_downsample, (int(size1/downsample_factor), int(size2/downsample_factor)))
    img_downsample = cv2.resize(img_downsample, (size1, size2))

    img_downsample_np = img_process(file_path, img_downsample)
    tensor_i = torch.from_numpy(img_downsample_np).type(torch.FloatTensor).cuda()
    feature_down = net(tensor_i)
    feature_down = feature_down.cpu()
    feature_down = feature_down.detach().numpy()[0]
    return (feature_down, img_cv, img_downsample, label)


def dataset_load(filename,zfile, whether_align=True):
    """
    最后一个参数表示是否需要进行剪裁，默认需要
    """
    # name: sphere20a:train
    # filename: ../../../data/CASIA.zip:0663546/027.jpg       4029    98      101     149     108     108     145     100     166     148     177
    # pindex: 5
    # <class 'dataset.'>
    # <zipfile.ZipFile filename='../../../data/CASIA.zip' mode='r'>
    
    position = filename.rfind('.zip:')
    zipfilename = filename[0:position+4]
    nameinzip = filename[position+5:]
    split = nameinzip.split('\t')
    nameinzip = split[0]
    classid = int(split[1])
    src_pts = []
    for i in range(5):
        src_pts.append([int(split[2*i+2]),int(split[2*i+3])])

    data = np.frombuffer(zfile.read(nameinzip),np.uint8)
    img = cv2.imdecode(data,1)

    if whether_align is True:
        img = alignment(img,src_pts)
        # 如果是处理lr图片，下面的操作也都在数据预处理中进行过
        # if ':train' in name:
        #     if random.random()>0.5: img = cv2.flip(img,1)
        #     if random.random()>0.5:
        #         rx = random.randint(0,2*2)
        #         ry = random.randint(0,2*2)
        #         img = img[ry:ry+112,rx:rx+96,:]
        #     else:
        #         img = img[2:2+112,2:2+96,:]
        img = img[2:2+112,2:2+96,:]
        img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    else:
        img = img.transpose(2, 0, 1).reshape((1,3,28, 24))
    img = ( img - 127.5 ) / 128.0   # 这一步问什么要这样一直没搞懂，以及注意在应对lr的数据预处理时有没有搞错
    label = np.zeros((1,1),np.float32)
    # img = torch.from_numpy(img)
    # label[0,0] = classid
    return (img,classid)


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def generator_noisy_dict_from_file(landmark, zfile, class_num, net,\
                                   noise_type="sp_noise", prob=0):
    tensor_dict = {}
    picked_dict = {}
    for i in range(class_num):
        tensor_dict[i] = None
        picked_dict[i] = []
    with open(landmark, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            total_path = testset_path + ":" + line
            if noise_type=="sp_noise":
                img_tensor, _, _, label = add_noise(total_path, zfile, prob)
    
            if tensor_dict[label] is None:
                tensor_dict[label] = img_tensor
            else:
                tensor_dict[label] = np.vstack((tensor_dict[label], img_tensor))
            picked_dict[label].append(total_path)
    f.close()
    return picked_dict, tensor_dict



def generator_numpy_dict_from_file(landmark, zfile, class_num, net, whether_resize=False, factor=1):
    """
    从文件列表中将数据集中相应的图片读取成numpy格式，同时根据class_num
    将每个类对应的numpy张量，存储为dict格式
    """
    tensor_dict = {}
    picked_dict = {}

    for i in range(class_num):
        tensor_dict[i] = None
        picked_dict[i] = []
    
    with open(landmark, 'r') as f:
        f_lines = f.readlines()
        for line in f_lines:
            total_path = testset_path + ":" + line

            if whether_resize is True:
                img_tensor, _, _, label = down_sample(total_path, zfile, factor, net)
            else:
                img_tensor, label = dataset_load(total_path, zfile)
    
            if tensor_dict[label] is None:
                tensor_dict[label] = img_tensor
            else:
                tensor_dict[label] = np.vstack((tensor_dict[label], img_tensor))
            picked_dict[label].append(total_path)
    
    f.close()

    if whether_resize:
        return tensor_dict

    for i in range(class_num):
        # print("test: original & downsize:", np.shape(tensor_dict[i]))
        tensor_i = torch.from_numpy(tensor_dict[i]).type(torch.FloatTensor).cuda()
        print("test:", tensor_i.size())
        feature = net(tensor_i)
        print("test:", feature.size())
        feature = feature.cpu()
        tensor_dict[i] = feature.detach().numpy()
    return picked_dict, tensor_dict


net = getattr(net_sphere_toy,args.net)()
net.load_state_dict(torch.load(args.model_file))
net.cuda()
net.feature = True

if __name__ == "__main__":
    testset_path = args.dataset_name_hr
    zfile = zipfile.ZipFile(testset_path)
    class_num = 10
    color_param      = ["#fd5956", "#ffd8b1", "#fdee73", "#2fef10", "#6488ea", "#13eac9", "#d94ff5", "#95d0fc", "#fb5ffc", "#fd4659"]
    color_param_mark = ["#9a0200", "#b66325", "#c9b003", "#789b73", "#0343df", "#137e6d", "#4e0550", "#3b719f", "#751973", "#770001"]

    # 画图
    import matplotlib.pyplot as plt
    plt.figure()
    
    

    # print(net)
    picked_dict, tensor_dict = generator_numpy_dict_from_file(\
                                  args.landmark, zfile, class_num, net)
    print("test:")
    for i in range(class_num):
        a, b = np.shape(tensor_dict[i])
        print(np.shape(tensor_dict[i]))
        for j in range(a):
            plt.plot(tensor_dict[i][j, 0], tensor_dict[i][j, 1],  color=color_param[i], marker='.')
    
    
    resize_factor_list = [4]
    # resize_factor_list = [4, 7, 10]
    for fig, factor in enumerate(resize_factor_list):
        tensor_dict_resize = generator_numpy_dict_from_file(\
                               args.landmark, zfile, class_num, net,\
                               True, factor=factor)
        fig_num = 220 + fig + 2
        plt.subplot(fig_num)
        # 先画出原始的点云
        for i in range(class_num): 
            a, b = np.shape(tensor_dict[i])
            for j in range(a):
                plt.plot(tensor_dict[i][j, 0], tensor_dict[i][j, 1],  color=color_param[i], marker='.')
    
        # 再画出新的点云
        for i in range(class_num):
            a, b = np.shape(tensor_dict[i])
            for j in range(a):
                plt.plot(tensor_dict_resize[i][j, 0], tensor_dict_resize[i][j, 1],  color=color_param_mark[i], marker='.')
    
    
    plt.subplot(221)
    for i in range(class_num):
        a, b = np.shape(tensor_dict[i])
        for j in range(a):
            plt.plot(tensor_dict[i][j, 0], tensor_dict[i][j, 1],  color=color_param[i], marker='.')


    transform_net = transform_net(f_dimension=2, \
            mid_dimension=args.mid_dimen, mid_num=args.mid_num)

    plt.subplot(223)
    for i in range(class_num):
        a, b = np.shape(tensor_dict[i])
        for j in range(a):
            plt.plot(tensor_dict[i][j, 0], tensor_dict[i][j, 1],  color=color_param[i], marker='.')

    transform_dict = {}
    for i in range(class_num):
        tensor_i = torch.from_numpy(tensor_dict[i]).type(torch.FloatTensor)# .cuda()
        feature = transform_net(tensor_i)
        feature = feature.cpu()
        transform_dict[i] = feature.detach().numpy()
    for i in range(class_num):
        a, b = np.shape(tensor_dict[i])
        for j in range(a):
            plt.plot(tensor_dict[i][j, 0], tensor_dict[i][j, 1],  color=color_param_mark[i], marker='.')
        


    plt.show()
