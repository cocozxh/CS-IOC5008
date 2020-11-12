# Fine-Grained Car Image Classification

Code for homework 1 in the Selected Topics in Visual Recognition using Deep Learning.
I choose the ResNet-50 as my backbone network and apply the transfer learning to speed up the training process.
## Catalog
- [Fine-Grained Car Image Classification](#fine-grained-car-image-classification)
  - [Catalog](#Catalog)
  - [Installation](#Installation)
    - [Dataset](#Dataset)
    - [Requirements](#Requirements)
    - [Pretrained model](#Pretrained-model)
  - [Dataset preparation](#Dataset-Preparation)
    - [Training data](#training-data)
    - [Split dataset](#split-dataset)
  - [Train](#train)
    - [Data augmentation](#data-augmentation)
    - [Train model](#train-model)
  - [Test](#test)

## Installation
### Dataset
  - [Link](https://www.kaggle.com/c/cs-t0828-2020-hw1/data)
### Requirements
- Python >= 3.6
- PyTorch >= 1.3.0
### Pretrained model
  -  [Link](https://baidu.com/) 

## Dataset preparation
After downloading the image data, the data directory is structured as:
```
data
  +- training data
  |  +- training data
  |  |  +- 000001.jpg
  |  |  +- 000002.jpg
  |  |  ...
  +- testing data
  |  +- testing data
  |  |  +- 000004.jpg
  |  |  +- 000005.jpg
  |  |  ...
  +- training_labels.csv
```
### Training data
All categories of training data are in one directory, so we should prepare the training data. 
```
$ python dataset.py
```
Then the training data directory is structured as:
```
training data
  +- Acura Integra Type R 2001
  |  +- 000406.jpg
  |  +- 000408.jpg
  |  +- ...
  +- Acura RL Sedan 2012
  |  +- 000091.jpg
  |  +- 000092.jpg
  |  +- ...
  +- Acura TL Sedan 2012
  |  +- 000154.jpg
  |  +- 000155.jpg
  |  +- ...
  |  ...
```
### Split dataset
I randomly sample 10% data from each class of training data to construct the validation set.
```
$ python train_validation.py
```
## Train
### Data augmentation
The best solution I found resizes the image to (448, 448) with RandomRotation, RandomHorizontalFlip and Normalization (mean=(0.4705, 0.4597, 0.4545), std=(0.2648, 0.2644, 0.2734)).
### Train model
To train models, run following commands.
```
$ python train.py --pre_train --rotation --resize 448 --epoch 60
```
The pretrained model on the ImageNet is loaded. The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | Testing accuracy
------------ | ------------- | ------------- | ------------- | ------------- | -------------
resnet50 | 1x TitanXp | 448 | 60 | 3.5 hours | 93.24%
resnet50 | 1x TitanXp | 224 | 60 | 1.1 hours | 87.38%
resnet50 | 1x TitanXp | 32 | 210 | 3.2 hours | ～10%

## Test
Run following commands to generate the submission.csv file of the testing results.
```
$ python test.py
```
