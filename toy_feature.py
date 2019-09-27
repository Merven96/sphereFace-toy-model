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
import scipy.io as io

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
# import net_sphere
import net_sphere_toy

"""
本程序的作用在于将args.dataset对应的文件夹中的人脸图片依次转换成特征向量,
并保存成.mat格式文件
"""

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
parser.add_argument('--batch_size', type=int, default=3)
# 11, 13, 43
parser.add_argument('--save_name', type=str, default="./toy_feature_train.mat")
parser.add_argument('--use_gpu', type=str, default='0', help='gpu')
parser.add_argument('--landmark', type=str)
args = parser.parse_args()

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
    # print("test2:", np.shape(img))
    if whether_align is True:
        img = alignment(img,src_pts)
        img = img[2:2+112,2:2+96,:]
        img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    else:
        img = img.transpose(2, 0, 1).reshape((1,3,28, 24))
    img = ( img - 127.5 ) / 128.0   # 这一步问什么要这样一直没搞懂，以及注意在应对lr的数据预处理时有没有搞错
    label = np.zeros((1,1),np.float32)
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




print("test: ", args.save_name)

os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu 
predicts=[]

net = getattr(net_sphere_toy,args.net)()

net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True
net.requires_grad = True


landmark = {}
with open(args.landmark) as f:
    landmark_lines = f.readlines()
num_person = len(landmark_lines)  # number of the total images

img = []
feature_list = []
    
zfile = zipfile.ZipFile(args.dataset)
batch_num = int(num_person / args.batch_size)
for i in range(batch_num):
    img_batch = []
    for j in range(args.batch_size):
        line = landmark_lines[i*args.batch_size + j]
        filename = args.dataset + ":" + line
        img_numpy, _ = dataset_load(filename, zfile)


        img_batch.append(img_numpy)

    img_list = np.vstack(img_batch)

    print(i ,'/', batch_num)
    print(np.shape(img_list))

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
