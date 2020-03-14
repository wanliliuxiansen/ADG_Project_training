# coding=utf-8
import os

import numpy as np

import torch

from skimage.color import rgb2lab, rgb2gray

from torchvision import datasets, transforms

class GrayscaleImageFolder(datasets.ImageFolder):
    '''
    Custom images folder, which converts imge to garyscale before loading
    自定义图像文件夹，在载入前将彩色图片转换为灰度图
    '''
    def __getitem__(self, index):

        global img_original
        global img_ab
        if self.transform is not None:

            path, target = self.imgs[index]
            img = self.loader(path)
            img = self.transform(img)
            img_original = np.asarray(img)#array仍会copy出一个副本，占用新的内存，但asarray不会。

            img_lab = rgb2lab(img_original)#rgb转化成lab
            img_lab = (img_lab + 128) / 255#将数值区间转化为0-255再转化为0-1区间内
            img_ab = img_lab[:, :, 1:3]#取三通道后两个值
            img_ab = torch.from_numpy(img_ab.transpose((2, 1, 0))).float()#转置

            img_original = rgb2gray(img_original)#转化为灰度图
            img_original = torch.from_numpy(img_original.T).unsqueeze(0).float()#增加一个0维度

            return img_original, img_ab, target

class open_val():
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
    def get_img_loader(self):

        transform = transforms.Compose([
            transforms.Resize(224, interpolation=2),
            transforms.CenterCrop(224),
        ])

        val_imgefolder = GrayscaleImageFolder(self.path, transform)

        val_loader = torch.utils.data.DataLoader(
            val_imgefolder,
            batch_size=self.batch_size,
            shuffle=False
        )
        return  val_loader