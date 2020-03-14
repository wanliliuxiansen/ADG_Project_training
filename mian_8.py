# coding=utf-8
import os

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np

from skimage.color import lab2rgb

from PIL import Image

import open as open
import function_4_0 as function

'''
检测GPU是否存在
'''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
引入函数
'''
D = function.Discriminator(3).to(DEVICE)
G = function.Generator().to(DEVICE)
'''
导入模型
'''
D.load_state_dict(torch.load(r'model2/model_D327680.pkl'))
G.load_state_dict(torch.load(r'model2/model_G24576.pkl'))
'''
损失函数
'''
criterion_BCE = nn.BCELoss()
criterion_MSE = nn.MSELoss()
criterion_L1Loss = nn.L1Loss()
'''
定义优化算法
'''
D_opt = torch.optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=0.002)#00 , betas=(0.5, 0.999)
'''
一些参数定义
'''
batch_size = 4

max_epoch = 2000

n_critic = 25



'''
输入数据集
'''
train = open.open_train('image_c/train', batch_size)
train_loader = train.get_img_loader()

val = open.open_val('image_c/val', batch_size)
val_loader = val.get_img_loader()

def save_image(gary, ab, step):
    G_ab = ab
    test_Lab = np.ones((284, 284, 3))
    test_Lab[..., 0] = np.array(gary.cpu().clone())[0][0] * 100
    G_ab = G_ab.cpu().data.numpy()[0]
    G_ab = np.asarray(G_ab).transpose((1, 2, 0))
    G_ab[G_ab < 0] = 0
    G_ab[G_ab > 1] = 1
    G_ab = G_ab * 255 - 128
    test_Lab[..., 1:3] = G_ab[..., 0:2]
    test_Lab = test_Lab.transpose((1, 0, 2))
    test_rgb = lab2rgb(test_Lab) * 255
    test_rgb = np.array(list(test_rgb), dtype=np.uint8)
    new_image = Image.fromarray(test_rgb)
    new_image.save('output/' + str(step) + '.jpg')

'''
训练
'''
def train(train_loader, D, G, criterion_MSE, criterion_L1Loss, D_opt, G_opt, DEVICE):
    global G_loss
    global D_loss
    step = 0
    D_labels = torch.ones(60, 60).to(DEVICE)  # Discriminator Label to real
    D_fakes = torch.zeros(60, 60).to(DEVICE)  # Discriminator Label to fake

    for epoch in range(max_epoch):
        for idx, (input_gary, input_ab, target) in enumerate(train_loader):
            if DEVICE:
                input_gary, input_ab, target = input_gary.cuda(), input_ab.cuda(), target.cuda()

            z_outputs = G(input_gary)
            input_all = torch.cat((input_gary, z_outputs), 1)

            real_out = torch.cat((input_gary, input_ab), 1)
            G_z_loss = criterion_L1Loss(input_all, real_out)  # 适当调大

            x_outputs = D(input_all)[0][0]
            G_x_loss = criterion_MSE(x_outputs, D_labels)

            G_loss = G_x_loss + G_z_loss

            G.zero_grad()
            G_loss.backward()
            G_opt.step()

            if idx % n_critic == 0 and idx != 0:
                input_all = torch.cat((input_gary, input_ab), 1)
                x_outputs = D(input_all)[0][0]

                D_x_loss = criterion_MSE(x_outputs, D_labels)

                input_all = torch.cat((input_gary, G(input_gary)), 1)
                z_outputs = D(input_all)[0][0]

                D_z_loss = criterion_MSE(z_outputs, D_fakes)
                D_loss = D_x_loss + D_z_loss

                D.zero_grad()
                D_loss.backward()
                D_opt.step()


            if idx % 25 == 0 and idx != 0:
                print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(
                    epoch, max_epoch,
                    idx * 4,
                    D_loss.item(),
                    G_loss.item()
                    )
                )
            if idx % 2048 == 0 and step != 0:
                torch.save(D.state_dict(), r'modelcs_0/model_D' + str(4 * step) + '.pkl')
                torch.save(G.state_dict(), r'modelcs_0/model_G' + str(4 * step) + '.pkl')
            step += 1

def validate(val_loader, G, DEVICE):
    for idx, (input_gray, input_ab, target) in enumerate(val_loader):
        if DEVICE:
            input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()
        y_hat = G(input_gray).to(DEVICE)
        save_image(input_gray, y_hat, idx)

def Preliminary(train_loader, D, criterion_MSE, DEVICE, max_epoch):
    D_labels = torch.ones(batch_size, 1).to(DEVICE)  # Discriminator Label to real

    for epoch in range(max_epoch):
        for idx, (input_gary, input_ab, target) in enumerate(train_loader):
            if DEVICE:
                input_gary, input_ab, target = input_gary.cuda(), input_ab.cuda(), target.cuda()
            input_all = torch.cat((input_gary, input_ab), 1)
            x_outputs = D(input_all)

            D_loss = criterion_MSE(x_outputs, D_labels)
            D.zero_grad()
            D_loss.backward()
            D_opt.step()
            if idx % 10 == 0:
                print('Epoch: {}/{}, Step: {}, D Loss: {}'.format(
                    epoch, max_epoch,
                    idx * 128,
                    D_loss.item(),
                    )
                )

train(train_loader, D, G, criterion_MSE, criterion_L1Loss, D_opt, G_opt, DEVICE)