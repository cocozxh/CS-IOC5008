# Fine-Grained Car Image Classification

Code for homework 1 in the Selected Topics in Visual Recognition using Deep Learning.

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

## Installation
### Dataset
  - [Link](https://www.kaggle.com/c/cs-t0828-2020-hw1/data)
### Requirements
- Python >= 3.7
- PyTorch >= 1.4.0
### Pretrained model
  -  [Link](https://baidu.com/) 

## Dataset preparation
After downloading the image data, the data directory is structured as:
```
data
  +- training data
  |  +- training data
  |    |  +- 000001.jpg
  |    |  +- 000002.jpg
  +- testing data
  |  +- testing data
  |    |  +- 000004.jpg
  |    |  +- 000005.jpg
  +- training_labels.csv
```
### Training data
All categories of training data are in one directory, so we should prepare the training data using [dataset.py](https://github.com/cocozxh/CS-IOC5008/blob/main/HW1/dataset.py). Then the training data directory is structured as:
```
training data
  +- Acura Integra Type R 2001
  |  +- 000406.jpgra
  |  +- 000408.jpg
  +- Acura RL Sedan 2012
  |  +- 000091.jpg
  |  +- 000092.jpg
  +- Acura TL Sedan 2012
  |  +- 000154.jpg
  |  +- 000155.jpg
```
### Split dataset
I randomly sample 10% data from each class of training data to construct the validation set using [train_validation.py](https://github.com/cocozxh/CS-IOC5008/blob/main/HW1/train_validation.py)

