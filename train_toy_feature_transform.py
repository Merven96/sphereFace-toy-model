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

# from dataset_v2 import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere_toy
# import net_sphere

import time
import sys
sys.path.insert(0, "../")
from datasets import ImageDataset_two_dir_with_label

parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lr_dataset', default='../../../data/CASIA_ALIGN.zip', type=str)
parser.add_argument('--dataset_name_hr', type=str, default="../../../data/CASIA.zip", help='zip-file of hr dataset')
parser.add_argument('--dataset_name_lr', type=str, default="../../../data/CASIA_ALIGN.zip", help='zip-file of the lr dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--landmark', type=str, default="./data/casia_landmark.txt")
parser.add_argument("--use_gpu", type=str, default='1')


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


def dataset_load(name,filename,pindex,cacheobj,zfile, whether_align=True):
    """
    最后一个参数表示是否需要进行剪裁，默认需要
    """
    # name: sphere20a:train
    # filename: ../../../data/CASIA.zip:0663546/027.jpg       4029    98      101     149     108     108     145     100     166     148     177
    # pindex: 5
    # <class 'dataset.'>
    # <zipfile.ZipFile filename='../../../data/CASIA.zip' mode='r'>
    
    # print("name:", name)
    # print("filename:", filename)
    # print("pindex:", pindex)
    # print(cacheobj)
    # print(zfile)
    # time.sleep(30)

    
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
        if ':train' in name:
            if random.random()>0.5: img = cv2.flip(img,1)
            if random.random()>0.5:
                rx = random.randint(0,2*2)
                ry = random.randint(0,2*2)
                img = img[ry:ry+112,rx:rx+96,:]
            else:
                img = img[2:2+112,2:2+96,:]
        else:
            img = img[2:2+112,2:2+96,:]
        img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    else:
        img = img.transpose(2, 0, 1).reshape((1,3,28, 24))
    img = ( img - 127.5 ) / 128.0   # 这一步问什么要这样一直没搞懂，以及注意在应对lr的数据预处理时有没有搞错
    label = np.zeros((1,1),np.float32)
    label[0,0] = classid
    return (img,label)


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



def train(epoch,args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    
    lr_transforms = [ transforms.ToTensor(), ]
    hr_transforms = [ transforms.ToTensor(), ]
    dataset_new = ImageDataset_two_dir_with_label(args.dataset_name_hr, \
                                       args.dataset_name_lr, \
                                       args.landmark,\
                                       # "./data/casia_landmark.txt",\
                                       lr_transforms, hr_transforms)
    
    
    dataloader_new = DataLoader(dataset_new,\
                               batch_size=args.batch_size, \
                               shuffle=True, num_workers=args.n_cpu)
    
    # ds = ImageDataset(args.dataset,dataset_load,'data/casia_landmark.txt',name=args.net+':train',
    #     bs=args.bs,shuffle=True,nthread=6,imagesize=128, low_resolution_root=args.lr_dataset)
    while True:
        for i, imgs in enumerate(dataloader_new):
            if imgs is None: break

            img_squeeze = torch.squeeze(imgs['hr'][0])
            inputs = img_squeeze.float()
            targets = torch.from_numpy(imgs['hr'][1].numpy()).long()
            # print("test the size:", img_squeeze.size())
            # [32, 3, 112, 96]

            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            # print(type(outputs), type(targets))
            # print(outputs, targets)
            loss = criterion(outputs, targets)
            # lossd = loss.data[0]  # 用于打印显示结果
            lossd = loss.item()  # 用于打印显示结果
            loss.backward()
            optimizer.step()

            # train_loss += loss.data[0]
            train_loss += loss.item()
            outputs = outputs[0] # 0=cos_theta 1=phi_theta
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            printoneline(dt(),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f %.2f %d'
                % (epoch,train_loss/(batch_idx+1), 100.0*correct/total, correct, total, 
                lossd, criterion.lamb, criterion.it))
            batch_idx += 1
        save_model(net, '{}_{}_toy_v2_while.pth'.format(args.net,epoch))
        print('******** SAVE **********')


# net = getattr(net_sphere,args.net)()
net = getattr(net_sphere_toy,args.net)()
# net.load_state_dict(torch.load('sphere20a_0.pth'))
net.cuda()
criterion = net_sphere_toy.AngleLoss()


print('start: time={}'.format(dt()))
for epoch in range(0, 20):
    if epoch in [0, 1, 3, 5,10,15]:
        if epoch!=0: args.lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train(epoch,args)
    save_model(net, '{}_{}_toy01.pth'.format(args.net,epoch))
    print("save for {} epoch".format(epoch))
print('finish: time={}\n'.format(dt()))

