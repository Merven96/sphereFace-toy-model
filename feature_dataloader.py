import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import zipfile
import cv2
# from matlab_cp2tform import get_similarity_transform_for_cv2

class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        return_dict = {}
        if self.lr_transform is not None:
            img_lr = self.lr_transform(img)
            return_dict["lr"] = img_lr
        elif self.hr_transform is not None:
            img1 = self.hr_transform(img)
            return_dict["hr"] = img_hr
        return return_dict

    def __len__(self):
        return len(self.files)

class FeatureDataset(Dataset):
    def __init__(self, input_list, label_list):
        self.input_data = input_list
        self.label_data = label_list
        self.data_num = len(input_list)

    def __getitem__(self, index):
        input_feature = self.input_data[index % self.data_num, :]
        label_feature = self.label_data[index % self.data_num, :]
        # print("test in dataLoader")
        # print(input_feature, np.shape(input_feature))
        input_feature = np.expand_dims(input_feature, axis=0)
        # print(input_feature, np.shape(input_feature))
        return input_feature, label_feature

    def __len__(self):
        return self.data_num



class ImageDataset_two_dir(Dataset):
    def __init__(self, root_hr, root_lr, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.files_hr = sorted(glob.glob(root_hr + '/*.*'))
        self.files_lr = sorted(glob.glob(root_lr + '/*.*'))

    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)])
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)])

        # print("test:", np.shape(img_hr))  # (96, 112, 3)
        # time.sleep(30)

        return_dict = {}

        if self.lr_transform is not None:
            img_lr = self.lr_transform(img_lr)
            return_dict["lr"] = img_lr
        if self.hr_transform is not None:
            img_hr = self.hr_transform(img_hr)
            return_dict["hr"] = img_hr
        return return_dict

    def __len__(self):
        len1 = len(self.files_lr)
        len2 = len(self.files_hr)
        if not (len1==len2):
           print("warning: number(low_images) != number(hr_images)")
        return len1


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

def dataset_load(filename, zfile, whether_align=True):
    """
    最后一个参数表示是否需要进行剪裁，默认需要
    """
    # filename: ../../../data/CASIA.zip:0663546/027.jpg       4029    98      101     149     108     108     145     100     166     148     177
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
        # if random.random()>0.5: img = cv2.flip(img,1)
        # if random.random()>0.5:
        #     rx = random.randint(0,2*2)
        #     ry = random.randint(0,2*2)
        #     img = img[ry:ry+112,rx:rx+96,:]
        # else:
        #     img = img[2:2+112,2:2+96,:]
        img = img[2:2+112,2:2+96,:]
        img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    else:
        img = img.transpose(2, 0, 1).reshape((1,3,28, 24))
    img = ( img - 127.5 ) / 128.0   # 这一步问什么要这样一直没搞懂，以及注意在应对lr的数据预处理时有没有搞错
    label = np.zeros((1,1),np.float32)
    label[0,0] = classid
    img = torch.from_numpy(img)
    # print("test:", filename, " ", classid, nameinzip)
    return (img,label[0, 0])
    





class ImageDataset_two_dir_with_label(Dataset):
    def __init__(self, root_hr, root_lr, image_list_file, lr_transforms, hr_transforms):
        # root_hr, root_lr: 相应zip文件的路径
        # image_list_file: ./spherer_pytorch/data/lanemark.txt
        # 如果用上面定义的dataset_load，transform参数不应该有归一化
        self.file_list = []  # 注意为了要hr与lr对应，file_list只定义一个
        with open(image_list_file) as f:
            lines = f.readlines()
        for line in lines:
            self.file_list.append(line)
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.zfile_hr = zipfile.ZipFile(root_hr)
        self.zfile_lr = zipfile.ZipFile(root_lr)
        self.root_hr = root_hr
        self.root_lr = root_lr

    def __getitem__(self, index):
        filename =  self.file_list
        filename = filename[index % len(self.file_list)]
        return_dict = {}
        if self.lr_transform is not None:
            lr_return = dataset_load(self.root_lr + ':' + filename, self.zfile_lr, False)
            return_dict["lr"] = lr_return 
        if self.hr_transform is not None:
            hr_return = dataset_load(self.root_hr+ ':'  + filename, self.zfile_hr, True)
            return_dict["hr"] = hr_return 
        return return_dict

    def __len__(self):
        len1 = len(self.file_list)
        return len1


class ImageDataset_with_label_with_filename(Dataset):
    def __init__(self, root_hr, root_lr, image_list_file, lr_transforms, hr_transforms):
        # root_hr, root_lr: 相应zip文件的路径
        # image_list_file: ./spherer_pytorch/data/lanemark.txt
        # 如果用上面定义的dataset_load，transform参数不应该有归一化
        self.file_list = []  # 注意为了要hr与lr对应，file_list只定义一个
        with open(image_list_file) as f:
            lines = f.readlines()
        for line in lines:
            self.file_list.append(line)
        # self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.zfile_hr = zipfile.ZipFile(root_hr)
        # self.zfile_lr = zipfile.ZipFile(root_lr)
        self.root_hr = root_hr
        # self.root_lr = root_lr

    def __getitem__(self, index):
        filename =  self.file_list
        filename = filename[index % len(self.file_list)]
        return_dict = {}
        # if self.lr_transform is not None:
        #     lr_return = dataset_load(self.root_lr + ':' + filename, self.zfile_lr, False)
        #     return_dict["lr"] = lr_return 
        if self.hr_transform is not None:
            hr_return = dataset_load(self.root_hr+ ':'  + filename, self.zfile_hr, True)
            return_dict["hr"] = hr_return 
        return_dict["filename"] = filename
        return return_dict

    def __len__(self):
        len1 = len(self.file_list)
        return len1



def count_image_for_everyone(file_name="./casia_landmark.txt", total_number = 10574):
    """
    返回一个dict类imageNum_dict, imageNum_dict['n']代表第n个人的图片数
    image_fileNameList_dict['0'] = [0000045/001.jpg, 0000045/002.jpg,
                           0000045/003.jpg, 0000045/004.jpg, 0000045/005.jpg,
                           0000045/006.jpg, 0000045/007.jpg, 0000045/008.jpg,
                           0000045/009.jpg, 0000045/011.jpg, 0000045/012.jpg,
                           0000045/013.jpg, 0000045/015.jpg]
    一共10574个人
    """
    image_fileNameList_dict = {}  
    image_file_id_List_dict = {}    
    image_Person_Num_dict   = {} 
    for i in range(total_number):
        image_fileNameList_dict[str(i)] = [] 
        image_file_id_List_dict[str(i)] = []   
        image_Person_Num_dict[str(i)] = 0 


    with open(file_name, 'r') as f:
        f_lines = f.readlines()
    i = 0
    for line in f_lines:
        line = line.strip('\n').split('\t')
        person_i = line[1]
        image_fileNameList_dict[person_i].append(line[0])
        image_file_id_List_dict[person_i].append(i)
        image_Person_Num_dict[person_i] = image_Person_Num_dict[person_i] + 1 
        i = i + 1
    #在数据接口中使用第二个返回值
    return image_fileNameList_dict, image_file_id_List_dict, image_Person_Num_dict



class ImageDataset_two_dir_label_and_counter(Dataset):
    def __init__(self, root_hr, root_lr, image_list_file, lr_transforms, hr_transforms, similarity_dict=None):
        # root_hr, root_lr: 相应zip文件的路径
        # image_list_file: ./spherer_pytorch/data/lanemark.txt
        # 如果用上面定义的dataset_load，transform参数不应该有归一化
        # similarity_dict中用webface_match.py中生成
        self.file_list = []  # 注意为了要hr与lr对应，file_list只定义一个

        _, self.same_person_dict, _ = count_image_for_everyone(image_list_file)

        with open(image_list_file) as f:
            lines = f.readlines()
        for line in lines:
            self.file_list.append(line)
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.zfile_hr = zipfile.ZipFile(root_hr)
        self.zfile_lr = zipfile.ZipFile(root_lr)
        self.root_hr = root_hr
        self.root_lr = root_lr
        self.similarity_dict = similarity_dict

    def __getitem__(self, index):
        filename =  self.file_list
        filename = filename[index % len(self.file_list)]
        
        # index对应的人的编号，int为了方便比较
        index_id = int(filename.split('\t')[1])  # 注意和index的不同
        
        same_list = self.same_person_dict[str(index_id)]

        if index not in same_list:
            print("some error in dataloader")
            exit()

        if self.similarity_dict is None:
            # 下面找出compare-image的id  (compare_id, 以及对应的compare_filename)
            same_person_num = len(same_list)
            if same_person_num == 1: # 只有一张照片
                compare_filename = filename
            else:
                compare_id = index
                while True:
                    random_id = np.random.randint(0, same_person_num)
                    compare_id = same_list[random_id]
                    if not(compare_id == index):
                        compare_filename = self.file_list[compare_id % \
                                                    len(self.file_list)]
                        break
        else:
            index_compare_list = self.similarity_dict[index]

            # print("test the similarity_dict:", index, index_compare_list)
            # time.sleep(5)

            random_id = np.random.randint(0, len(index_compare_list))
            compare_id = index_compare_list[random_id]
            compare_filename = self.file_list[compare_id % \
                                               len(self.file_list)]


        return_dict = {}
        if self.lr_transform is not None:
            lr_return = dataset_load(self.root_lr + ':' + filename, self.zfile_lr, False)
            return_dict["lr"] = lr_return 

            compare_lr_return = dataset_load(self.root_lr + ':' \
                                  + compare_filename, self.zfile_lr, False)
            return_dict["compare_lr"] = compare_lr_return 

        if self.hr_transform is not None:
            hr_return = dataset_load(self.root_hr+ ':'  + filename, self.zfile_hr, True)
            return_dict["hr"] = hr_return 

            compare_hr_return = dataset_load(self.root_hr+ ':'  \
                                  + compare_filename, self.zfile_hr, True)
            return_dict["compare_hr"] = compare_hr_return 
        return return_dict

    def __len__(self):
        len1 = len(self.file_list)
        return len1

