# coding=utf-8
import os
import sys

import torch
import torch.nn as nn

import numpy as np

import open_image_2 as open
import function_4_0 as function
# import function_3_5 as function

import torchvision.transforms as T

from skimage.color import lab2rgb
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = function.Generator().to(DEVICE)

G.load_state_dict(torch.load(r'modelcs/model_G32768.pkl'))
#
# G.load_state_dict(torch.load(r'model3/model_G24576(1).pkl'))
# G.load_state_dict(torch.load(r'model2/model_G106496.pkl'))
# G.load_state_dict(torch.load(r'model50/model_G184320.pkl'))

def save_image(gary, ab):
    test_Lab = np.ones((128, 128, 3))
    L = gary.cpu().clone()
    test_Lab[..., 0] = np.array(L)[0][0] * 100

    G_ab = ab
    G_ab = G_ab.cpu().data.numpy()[0]
    G_ab = np.asarray(G_ab).transpose((1, 2, 0))
    G_ab[G_ab < 0] = 0
    G_ab[G_ab > 1] = 1
    G_ab = G_ab * 255 - 128

    test_Lab[..., 1:3] = G_ab[..., 0:2]
    test_Lab = test_Lab.transpose((1, 0, 2))

    test_rgb = lab2rgb(test_Lab) * 255
    test_rgb = np.array(list(test_rgb), dtype=np.uint8)
    print(test_rgb)
    new_image = Image.fromarray(test_rgb)
    new_image.save('a.jpg')

def validate(image, G, DEVICE):
    if DEVICE:
        image = image.cuda()
    y_hat = G(image).to(DEVICE)
    save_image(image, y_hat)

# batch_size = 1
aDir = ' '.join(sys.argv[1:])
im = Image.open(aDir)#固定路径
data = aDir.split('/')[-1]
# im = Image.open('out_gary/s_0.jpg')
im = torch.from_numpy(np.array(im) / 255).unsqueeze(0).unsqueeze(0).float()
# print(im.shape)
# val = open.open_val('image_c/val', batch_size)
# val_loader = val.get_img_loader()
validate(im, G, DEVICE)