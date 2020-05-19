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

def generate(Gs, Zs, images1, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=20):
    # if in_s is None:
    #     in_s = torch.full(images1[0].shape, 0, device=opt.device)
    # images_cur = []
    # x=0
    #
    # for G, Z_opt, real_curr, noise_amp in zip(Gs, Zs, images1, NoiseAmp): #从最底层开始
    #     pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2 #做一些规格计算的准备工作
    #     m = nn.ZeroPad2d(int(pad1))
    #     # nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
    #     # nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h
    #     nzx = (Z_opt.shape[2] ) * scale_v
    #     nzy = (Z_opt.shape[3] ) * scale_h
    #
    #     images_prev = images_cur
    #     images_cur = []
    #
    #     for i in range(0,num_samples,1): # for 五十张图像
    #         if n == 0: #产生噪声
    #             z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
    #             z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
    #             z_curr = m(z_curr)
    #         else: #产生噪声
    #             z_curr = functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
    #             # print("z_curr1:", end="")
    #             # print(z_curr.size())
    #             z_curr = m(z_curr)
    #             # print("z_curr2:", end="")
    #             # print(z_curr.size())
    #
    #         if images_prev == []:
    #             I_prev = m(in_s) # 若images_prev为空（第一次循环），则将输入的in_s作为I_prev
    #         else:
    #             I_prev = images_prev[i] # 继承并上采样上一级传过来的图像
    #             I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
    #             if opt.mode != "SR":  # 对 I_prev进行规格修改和上采样
    #                 I_prev = I_prev[:, :, 0:round(scale_v * images1[n].shape[2]), 0:round(scale_h * images1[n].shape[3])]
    #                 I_prev = m(I_prev)
    #                 I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
    #                 I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])
    #             else:
    #                 I_prev = m(I_prev)
    #
    #         if n < gen_start_scale:
    #             z_curr = Z_opt
    #
    #         #z_in = noise_amp * (z_curr) + real_curr # 噪声 = 噪声+test_image
    #         print("z_curr:", end="")
    #         print(z_curr.size())
    #         # z_in = noise_amp * (z_curr) + I_prev # 噪声 = 噪声+上一级图像
    #         # if x==0:
    #         #     z_in = m(z_in) #这一句要删去！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    #         # x+=1
    #         z_in = noise_amp * (z_curr)
    #         print("I_prev:", end="")
    #         print(I_prev.size())
    #         print("z_in:", end="")
    #         print(z_in.size())
    #         z_in += I_prev  # 噪声 = 噪声+上一级图像
    #         I_curr = G(z_in.detach(), I_prev)
    #
    #         if n == len(images1) - 1: #若层数已到达最顶层，则存储图像
    #             if opt.mode == 'train':
    #                 dir2save = '%s/training_result/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
    #             else:
    #                 dir2save = functions.generate_dir2save(opt)
    #             try:
    #                 os.makedirs(dir2save)
    #             except OSError:
    #                 pass
    #             if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "test"):
    #                 plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
    #         images_cur.append(I_curr) #更新上一级图像
    #     n += 1 #层数+1
    # return I_curr.detach()


    if in_s is None:
        in_s = torch.full(images1[0].shape, 0, device=opt.device)

    #prev = in_s
    count = 0
    pad_noise = 0
    pad_image = 0
    prev = torch.full(images1[0].shape, 0, device=opt.device)

    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    if opt.mode == 'train':
        dir2save = '%s/%s/gen_start_scale=%d' % (opt.out, opt.input_name, gen_start_scale)
    else:
        dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,images1,images1[1:],NoiseAmp):
        opt.bsz = real_curr.shape[0]
        bsz = opt.bsz
        opt.nzx = real_curr.shape[2]
        opt.nzy = real_curr.shape[3]
        z = functions.generate_noise([3, Z_opt.shape[2] , Z_opt.shape[3] ], device=opt.device)
        z = z.expand(real_curr.shape[0], 3, z.shape[2], z.shape[3])

        # prev = cycle_rec(Gs, Zs, images1, NoiseAmp, in_s, m_noise, m_image, opt, bsz)
        # prev = m_image(prev)
        prev = prev[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
        print("z:", end="")
        print(z.size())
        print("prev:", end="")
        print(prev.size())
        z_in = noise_amp*z+real_curr
        in_s = G(z_in.detach(),prev)

        count += 1
        plt.imsave('%s/image_at_scale_%d.png' % (dir2save,count), functions.convert_image_np(in_s.detach()), vmin=0,vmax=1)
    return in_s.detach()


