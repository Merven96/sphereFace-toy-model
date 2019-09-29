from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile
from IPython import embed
import time

import net_sphere
from net_sphere_toy import transform_net

import scipy.io as io

"""
本程序用于测试训练好的transform-network，主要通过读取args.dataset中的数据
该数据通常为低质量的人脸数据,且已经经过剪裁
args.dataset转换为特征向量(通常为LFW)
用transform network对特征向量进行变换
最后按照BLUFR格式存储mat文件
"""

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--dataset', default='../PyTorch-GAN/data/lfw_dataset/data_align/inter_lfw.zip', type=str)
parser.add_argument('--lfw_landmark', default='../PyTorch-GAN/implementations/git_srgan/sphereface_pytorch/data/BLUFR_image_list.txt')
parser.add_argument('--BLUFR', default='../PyTorch-GAN/implementations/git_srgan/sphereface_pytorch/data/BLUFR_image_list.txt')

parser.add_argument('--model','-m', default='../PyTorch-GAN/implementations/git_srgan/sphereface_pytorch/model/sphere20a_20171020.pth', type=str)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--use_gpu', type=str, default='0', help='gpu')

parser.add_argument('--mid_num', type=int, default=6)
parser.add_argument('--mid_dimen', type=int, default=512)
parser.add_argument('--feature_dimen', type=int, default=512)
parser.add_argument('--transform_model', type=str, default='./result/high_dimension_result/hightransform_6-midNum_512-midDimen-epoch_20.pth')

parser.add_argument('--save_name', type=str, default="transform_lfw_feature.mat")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu 
predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True
net.requires_grad = True


transform_net = transform_net(f_dimension=args.feature_dimen, \
                    mid_dimension=args.mid_dimen, mid_num=args.mid_num)


zfile = zipfile.ZipFile(args.dataset)
landmark = {}
with open(args.BLUFR) as f:
    landmark_lines = f.readlines()
num_person = len(landmark_lines)  # number of the total images

img = []
feature_list = []
    
batch_num = int(num_person / args.batch_size)
for i in range(batch_num):
    img_batch = []
    for j in range(args.batch_size):
        line = landmark_lines[i*args.batch_size + j].strip('\n')
        # l = line.replace('\n','').split('\t')


        l = line.split('_')

        person_name = l[:-1]
        # 从图片名中恢复出文件夹名(人名)
        dir_name = person_name[0]
        if len(person_name)>1:
            for kk in range(1, len(person_name)):
                dir_name = dir_name + '_' + person_name[kk]

        dir_name = dir_name + "/"
        
        # print("test:\n", dir_name + line)

        img = cv2.imdecode(np.frombuffer(zfile.read(dir_name + line),np.uint8),1)
        img = img.transpose(2, 0, 1).reshape((1,3,np.shape(img)[0], np.shape(img)[1]))

        img = (img - 127.5)/128.0

        img_batch.append(img)
    img_list = np.vstack(img_batch)
    img_list = Variable(torch.from_numpy(img_list).float(),volatile=True).cuda()
    output = net(img_list)
    output = output.to('cpu')
    # print("test:", np.shape(output))

    if transform_net is not None:
        output = transform_net(output)

    output = output.detach().numpy()
    feature_list.append(output)

final_feature = np.vstack(feature_list)
print(type(final_feature))
print(np.shape(final_feature))

mat_save = args.save_name
io.savemat(mat_save, {'name':final_feature})
