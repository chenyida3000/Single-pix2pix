from __future__ import print_function
import utils.functions
import models.model as models
import argparse
import os
from utils.imresize import imresize
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from models.training import *
from options.config import get_arguments
import utils.functions as functions
import torch.optim as optim


def generate(Gs, Zs, images1, images2, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=50):
    if in_s is None:
        in_s = torch.full(images2[0].shape, 0, device=opt.device)
    images_cur = []
    noise_amp = 0.05

    # opt.mode = 'load_trained_model'

    for real_curr in images1: #从最底层开始
        # print(n)
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(n / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(n / 4)), 128)
        G,Z_opt = functions.load_G(opt,n)

        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2 #做一些规格计算的准备工作
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
        nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1): # for 五十张图像
            if n == 0: #产生噪声
                z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
                z_curr = m(z_curr)
            else: #产生噪声
                z_curr = functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                # I_prev = m(in_s) # 若images_prev为空（第一次循环），则将输入的in_s作为I_prev
                I_prev = in_s
            else:
                I_prev = images_prev[i] # 继承并上采样上一级传过来的图像
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                # if opt.mode != "SR":  # 对 I_prev进行规格修改和上采样
                I_prev = I_prev[:, :, 0:round(scale_v * images2[n].shape[2]), 0:round(scale_h * images2[n].shape[3])]
                I_prev = m(I_prev)
                I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])
                # else:
                #     I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            # z_in = noise_amp * z_curr + real_curr #噪声 = 噪声+real图像
            # z_in = z_curr + real_curr  # 噪声 = 噪声+real图像
            z_in = noise_amp * z_curr+ real_curr
            I_curr = G(z_in.detach(), I_prev)


            if opt.mode == 'train':
                dir2save = '%s/training_result/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
            else:
                dir2save = functions.generate_dir2save(opt)
            try:
                os.makedirs(dir2save)
            except OSError:
                pass
            if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR"):
                # print('!!!!!!')
                # print(dir2save)
                plt.imsave('%s/%d.png' % (dir2save, n), functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                # save_image(denorm(I_curr.data.cpu()), '%s/%d.png' % (dir2save, n))
            # if n == len(images2) - 1: #若层数已到达最顶层，则存储图像

            images_cur.append(I_curr) #更新上一级图像
        n += 1 #层数+1
    return I_curr.detach()

# def init_G(opt):
#     # generator initialization
#     netG = models.GeneratorConcatSkip2CleanAddAlpha(opt).to(opt.device)
#     netG.apply(models.weights_init)
#     if opt.netG != '':
#         netG.load_state_dict(torch.load(opt.netG))
#     # print(netG)
#
#     return netG
    # if in_s is None:
    #     in_s = torch.full(reals[0].shape, 0, device=opt.device)
    # x_ab = in_s
    # # x_aba = in_s
    # count = 0
    # if opt.mode == 'train':
    #     dir2save = '%s/%s/gen_start_scale=%d' % (opt.out, opt.input_name, gen_start_scale)
    # else:
    #     dir2save = functions.generate_dir2save(opt)
    # try:
    #     os.makedirs(dir2save)
    # except OSError:
    #     pass
    # for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
    #     z = functions.generate_noise([3, Z_opt.shape[2] , Z_opt.shape[3] ], device=opt.device)
    #     z = z.expand(real_curr.shape[0], 3, z.shape[2], z.shape[3])
    #     x_ab = x_ab[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
    #     z_in = noise_amp*z+real_curr
    #     x_ab = G(z_in.detach(),x_ab)
    #
    #     # x_aba = G2(x_ab,x_aba)
    #     x_ab = imresize(x_ab.detach(),1/opt.scale_factor,opt)
    #     x_ab = x_ab[:,:,0:real_next.shape[2],0:real_next.shape[3]]
    #     # x_aba = imresize(x_aba.detach(),1/opt.scale_factor,opt)
    #     # x_aba = x_aba[:,:,0:real_next.shape[2],0:real_next.shape[3]]
    #     count += 1
    #     plt.imsave('%s/x_ab_%d.png' % (dir2save,count), functions.convert_image_np(x_ab.detach()), vmin=0,vmax=1)


