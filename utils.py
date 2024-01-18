import os

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from config import config

args = config()
cls = args.class_num

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class Custom_Dataset(Dataset):
    def __init__(self, root,label, transform, csv_path, mode):
        super().__init__()
        self.root = root
        self.label = label
        self.transform = transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df
        self.mode = mode

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']
        label = patience_info['label']
        file = os.listdir(self.root+'/'+file_name)

        if self.mode == 'train':
            img1 = Image.open(self.root + '/' + file_name + '/'+ file[0]).convert('RGB')
            img2 = Image.open(self.root + '/' + file_name + '/'+ file[1]).convert('RGB')
            label1 = Image.open(self.label + '/' + file_name + '/'+ file[0]).convert('L')
            seg_labels1 = np.array(label1)
            label2 = Image.open(self.label + '/' + file_name + '/' + file[1]).convert('L')
            seg_labels2 = np.array(label2)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                seg_labels1 = self.transform(torch.from_numpy(seg_labels1).type(torch.FloatTensor))
                seg_labels2 = self.transform(torch.from_numpy(seg_labels2).type(torch.FloatTensor))

            seg_labels = torch.cat((seg_labels1, seg_labels2), dim=0)
            img = torch.cat((img1,img2),dim=0)

        if self.mode == 'test':
            img1 = Image.open(self.root + '/' + file_name + '/' + file[0]).convert('RGB')
            img2 = Image.open(self.root + '/' + file_name + '/' + file[1]).convert('RGB')
            label1 = Image.open(self.label + '/' + file_name + '/' + file[0]).convert('L')
            seg_labels1 = np.array(label1)
            label2 = Image.open(self.label + '/' + file_name + '/' + file[1]).convert('L')
            seg_labels2 = np.array(label2)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                seg_labels1 = self.transform(torch.from_numpy(seg_labels1).type(torch.FloatTensor))
                seg_labels2 = self.transform(torch.from_numpy(seg_labels2).type(torch.FloatTensor))

            seg_labels = torch.cat((seg_labels1, seg_labels2), dim=0)
            img = torch.cat((img1, img2), dim=0)

        return {'imgs': img, 'labels': label, 'names': file_name, 'region':seg_labels}

    def get_cls_num(self):
        cls_num = [0] * cls
        for img in self.info['label']:
            cls_num[int(img)] += 1
        return cls_num

    def __len__(self):
        return len(self.info)

def get_dataset(imgpath,labelpath, csvpath, img_size, mode='train', keyword=None):
    train_transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        # AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        # AddBlur(kernel=3, p=0.5),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
            transform = train_transform
    elif mode == 'test':
        transform = test_transform

    dataset = Custom_Dataset(imgpath, labelpath, transform, csvpath, mode)

    return dataset

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix

def plot_conf(matrix,num_classes,labels):
    matrix = matrix
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 设置x轴坐标label
    plt.xticks(range(num_classes), labels, rotation=45)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), labels)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion matrix')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            # info = int(matrix[y, x]),round(int(matrix[y, x])/int(matrix.sum(axis=0)[x]),2)
            info = int(matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if int(matrix[y, x]) > thresh else "black")
    plt.tight_layout()
    plt.show()


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ct, ht, wt = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    inputs = torch.softmax(inputs, dim=1)

    # 计算dice loss
    tp = torch.sum(target * inputs, dim=(0, 2, 3))
    fp = torch.sum(inputs, dim=(0, 2, 3)) - tp
    fn = torch.sum(target, dim=(0, 2, 3)) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss
