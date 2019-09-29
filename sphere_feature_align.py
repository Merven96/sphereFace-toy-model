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

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from IPython import embed
import time

import scipy.io as io

"""
此代码用于对lfw数据集生成LFW-BLUFR协议要求格式的特征向量文件
***其中输入的数据未经过剪裁!!
"""

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
    parser.add_argument('--net','-n', default='sphere20a', type=str)
    parser.add_argument('--dataset', default='../../../data/lfw_dataset/lfw_original_align.zip', type=str)
    parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--save_name', type=str, default="lfw_feature.mat")
    parser.add_argument('--use_gpu', type=str, default='0', help='gpu')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu 
    predicts=[]
    net = getattr(net_sphere,args.net)()
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    net.feature = True
    net.requires_grad = True
    
    zfile = zipfile.ZipFile(args.dataset)
    
    landmark = {}
    with open('data/BLUFR_image_list.txt') as f:
        landmark_lines = f.readlines()
    num_person = len(landmark_lines)  # number of the total images


    landmark_align = {}
    with open('data/lfw_landmark.txt') as f:
        landmark_align_lines = f.readlines()
    for line in landmark_align_lines:
        l = line.replace('\n','').split('\t')
        landmark_align[l[0]] = [int(k) for k in l[1:]]

    
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

            img = alignment(cv2.imdecode(np.frombuffer(zfile.read(dir_name + line),np.uint8),1), landmark_align[dir_name + line])
            img = img.transpose(2, 0, 1).reshape((1,3,np.shape(img)[0], np.shape(img)[1]))
    
            img = (img - 127.5)/128.0
    
            img_batch.append(img)
        img_list = np.vstack(img_batch)
        img_list = Variable(torch.from_numpy(img_list).float(),volatile=True).cuda()
        output = net(img_list)
        output = output.to('cpu')
        output = output.detach().numpy()
        feature_list.append(output)
    
    final_feature = np.vstack(feature_list)
    print(type(final_feature))
    print(np.shape(final_feature))
    
    mat_save = args.save_name
    io.savemat(mat_save, {'name':final_feature})
