from __future__ import print_function
from os.path import join
import argparse
import torch
import torch.nn as nn
import time
import math
from torch.autograd import Variable
from PIL import Image
import os

from torchvision.transforms import ToTensor
import numpy as np

# Demonstration settings
parser = argparse.ArgumentParser(description='VDSR PyTorch Demonstration')
parser.add_argument('--input_image', type=str,
                    required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True,
                    help='model file to use')
parser.add_argument('--output_filename', type=str,
                    help='where to save the output image')
parser.add_argument('--scale_factor', type=float,
                    help='factor by which super resolution needed')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
opt = parser.parse_args()
opt.gpuids = list(map(int, opt.gpuids))
print(opt)
model_name = opt.model
model = torch.load("model/"+model_name)

img_paths = os.listdir(opt.input_image)
for ip in img_paths:
    img = Image.open(opt.input_image+ip).convert('RGB')
    # y, cb, cr = img.split()
    # print(y)

    img = img.resize((int(img.size[0]*opt.scale_factor),
                        int(img.size[1]*opt.scale_factor)), Image.BICUBIC)
    img1 = img.convert('RGB')
    # img1.save('0000.png')
    # print(img.size) # (720, 576)
    img2 = np.asarray(img1)
    # print(img2.shape) # (576, 720, 3)


    input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])


    torch.cuda.set_device(opt.gpuids[0])
    with torch.cuda.device(opt.gpuids[0]):
        model = model.cuda()
    if opt.cuda:
        input = input.cuda()

    start_time = time.time()
    out = model(input)
    elapsed_time = time.time() - start_time
    print("===> It takes {:.4f} seconds.".format(elapsed_time))
    out = out.cpu()

    # print('out shape is {}'.format(out.shape)) # [1, 3, 576, 720]

    out_img_y = out.data[0].numpy()
    # print('out_img_y {}'.format(out_img_y.shape)) # [3, 576, 720]
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    # print('out_img_y {}'.format(out_img_y.shape)) # (3, 576, 720)
    out_img_y = out_img_y.transpose(1, 2, 0)
    out_img_y = Image.fromarray(out_img_y.astype('uint8')).convert('RGB')
    # out_img_y = Image.fromarray(np.uint8(out_img_y)).convert('RGB')
    print('out_img_y {}'.format(out_img_y.size))


    out_img_y.save('/output/'+ip)
    print('output image saved to ', '000.pngs')
