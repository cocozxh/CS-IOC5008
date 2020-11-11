import torch
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import csv

from model import *

path = '/data/zihaosh/car_train_val_test/data/'
train_save_path = path + 'train/'
path_labels = path + 'training_labels.csv'
path_test = path + 'testing_data/'
path_load_model = '/output/'

transforms_img = torchvision.transforms.Compose([
    torchvision.transforms.Resize([448, 448]),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4705, 0.4597, 0.4545), (0.2648, 0.2644, 0.2734))])

train_dataset = torchvision.datasets.ImageFolder(train_save_path,transform=transforms_img)
CLASS = train_dataset.class_to_idx

class TestDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, img_dir, transform=None):
        """
        Args:
        - data_dir: Directory containing dataset files
        """
        super(TestDataset, self).__init__()

        self.img_dir = img_dir
        self.transforms = transform
        self.image_data = []
        self.ids = []

        image_files = os.listdir(img_dir)
        self.num_seq = len(image_files)
        a = 0
        for file in image_files:
            input_img_path = self.img_dir + file
            index = file.split('.')[0]
#             print(a)
            a += 1
            self.image_data.append(input_img_path)
            self.ids.append(index)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        img = Image.open(self.image_data[index])
        if img.mode=='L':
            img = img.convert('RGB')
        id = self.ids[index]
        if self.transforms:
            img = self.transforms(img)
        return (img, id)
    
    
    
print('*'*50)
print('Load the test data')
transforms_imag=torchvision.transforms.Compose([
    torchvision.transforms.Resize([448,448]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4686,0.4588,0.4544), (0.2642,0.2641,0.2734))])
mydata = TestDataset(path_test, transform=transforms_imag)

test_loader = DataLoader(mydata, batch_size=2, shuffle=True, num_workers=2)


device = torch.device("cuda")


print('*'*50)
print('Initialize the model')
# 模型定义-ResNet
model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 196)
model_for_hw1 = model_ft.to(device)
##############################
model_for_hw1.load_state_dict(torch.load('net_059.pth'))

print('*'*50)
print("Waiting Test!")
csv_path = '/output/result_resnet50_best.csv'
with open(csv_path, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['id','label'])

a = 0
# test
with open(csv_path, 'a', newline='') as file:
    with torch.no_grad():
        model_for_hw1.eval()
        for data in test_loader:
            images, ids = data
            images = images.to(device)
            outputs = model_for_hw1(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(predicted.shape[0]):
                print(a)
                a += 1
                cls = list(CLASS.keys())[list(CLASS.values()).index(int(predicted[i]))]
                if cls == 'AAAA':
                    cls = 'Ram C/V Cargo Van Minivan 2012'
                row = [ids[i], cls]
                csvwriter = csv.writer(file)
                csvwriter.writerow(row)
