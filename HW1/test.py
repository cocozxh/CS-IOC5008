import torch
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
import csv

path = '/data/'
train_save_path = path + 'train/'
path_labels = path + 'training_labels.csv'
path_test = path + 'testing_data/testing_data/'
path_load_model = '/output/'

transforms_img = torchvision.transforms.Compose([
    torchvision.transforms.Resize([448, 448]),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4705, 0.4597, 0.4545),
                                     (0.2648, 0.2644, 0.2734))])

train_dataset = torchvision.datasets.ImageFolder(train_save_path,
                                                 transform=transforms_img)
CLASS = train_dataset.class_to_idx


class TestDataset(Dataset):
    def __init__(
            self, img_dir, transform=None):
        super(TestDataset, self).__init__()

        self.img_dir = img_dir
        self.transforms = transform
        self.image_data = []
        self.ids = []

        image_files = os.listdir(img_dir)
        self.num_seq = len(image_files)
        for file in image_files:
            input_img_path = self.img_dir + file
            index = file.split('.')[0]
            self.image_data.append(input_img_path)
            self.ids.append(index)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        img = Image.open(self.image_data[index])
        if img.mode == 'L':
            img = img.convert('RGB')
        id = self.ids[index]
        if self.transforms:
            img = self.transforms(img)
        return img, id


print('*' * 50)
print('Load the test data')
transforms_imag = torchvision.transforms.Compose([
    torchvision.transforms.Resize([448, 448]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4686, 0.4588, 0.4544),
                                     (0.2642, 0.2641, 0.2734))])
mydata = TestDataset(path_test, transform=transforms_imag)

test_loader = DataLoader(mydata, batch_size=2, shuffle=True, num_workers=2)

device = torch.device("cuda")

print('*' * 50)
print('Initialize the model')
model0 = models.resnet50(pretrained=False)
num_ftrs = model0.fc.in_features
model0.fc = nn.Linear(num_ftrs, 196)
model_hw1 = model0.to(device)
model_hw1.load_state_dict(torch.load('net_059.pth'))

print('*' * 50)
print("Waiting Test!")
csv_path = '/output/submission.csv'
with open(csv_path, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['id', 'label'])

# test
with open(csv_path, 'a', newline='') as file:
    with torch.no_grad():
        model_hw1.eval()
        for data in test_loader:
            images, ids = data
            images = images.to(device)
            outputs = model_hw1(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(predicted.shape[0]):
                predict = int(predicted[i])
                cls = list(CLASS.keys())[list(CLASS.values()).index(predict)]
                if cls == 'AAAA':
                    cls = 'Ram C/V Cargo Van Minivan 2012'
                row = [ids[i], cls]
                csvwriter = csv.writer(file)
                csvwriter.writerow(row)
