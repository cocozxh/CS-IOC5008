import torch
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--de', default='cuda')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--data', default='/data/zihaosh/car_train_val_test/data/')
parser.add_argument('--resize', type=int, default=448,
                    help='resize the image b4 input')
parser.add_argument('--pre_train', action="store_true", default=False,
                    help='use the pre train model')
parser.add_argument('--rotation', action="store_true", default=False,
                    help='use the rotation to preprocess the image')

args = parser.parse_args()
print(args)

path = args.data
train_save_path = path + 'train/'
val_save_path = path + 'validation/'
path_labels = path + 'training_labels.csv'
path_test = path + 'test/'
path_load_model = '/output/'

device = torch.device(args.de)

print('*' * 50)
print('Load the train data')
if args.rotation:
    transforms_img = torchvision.transforms.Compose([
        torchvision.transforms.Resize([args.resize, args.resize]),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4705, 0.4597, 0.4545),
                                         (0.2648, 0.2644, 0.2734))])
else:
    transforms_img = torchvision.transforms.Compose([
        torchvision.transforms.Resize([args.resize, args.resize]),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4705, 0.4597, 0.4545),
                                         (0.2648, 0.2644, 0.2734))])

train_dataset = torchvision.datasets.ImageFolder(train_save_path,
                                                 transform=transforms_img)
val_dataset = torchvision.datasets.ImageFolder(val_save_path,
                                               transform=transforms_img)
CLASS = train_dataset.class_to_idx

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True, num_workers=2)

print('*' * 50)
print('Initialize the model')
# 模型定义-ResNet
model0 = models.resnet50(pretrained=args.pre_train)
num_ftrs = model0.fc.in_features
model0.fc = nn.Linear(num_ftrs, 196)
net_hw1 = model0.to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net_hw1.parameters()),
                       lr=args.lr, weight_decay=args.wd)
# 训练
if __name__ == "__main__":
    print("Start Training!")
    with open("/output/val.txt", "w")as f:
        with open("/output/log.txt", "w")as f2:
            for epoch in range(args.epoch):
                print('\nEpoch: %d' % (epoch + 1))
                net_hw1.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = net_hw1(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    if (i + 1 + epoch * length) % 100 == 0:
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * length),
                                 sum_loss / (i + 1), 100. * correct / total))

                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length),
                                sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下validation准确率
                print("Waiting Validate!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        net_hw1.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net_hw1(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('Validation accuracy is: %.3f%%'
                          % (100. * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net_hw1.state_dict(), '%s/net_%03d.pth'
                               % ('/output', epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

    print("Training Finished!")
