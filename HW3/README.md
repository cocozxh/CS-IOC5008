# Digit Detection

Code for homework 2 in the Selected Topics in Visual Recognition using Deep Learning.
I choose the YOLOv4 as my network and apply the transfer learning to speed up the training process.
## Catalog
- [Digit Detection](#digit-detection)
  - [Catalog](#Catalog)
  - [Installation](#Installation)
    - [Dataset](#Dataset)
    - [Requirements](#Requirements)
    - [Pretrained model](#Pretrained-model)
  - [Dataset preparation](#Dataset-Preparation)
  - [Train](#train)
    - [Data pre-process](#data-pre-process)
    - [Train model](#train-model)
  - [Test](#test)
  - [Speed](#speed)

## Installation
### Dataset
The dataset for this homework is here:
  - [Link](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl)
### Requirements
- Python >= 3.6
- PyTorch >= 1.3.0
### Pretrained model
The final model I traiined is here:
  -  [Final Model](https://pan.baidu.com/s/1TO-wO79aJyK5c_OSGPKS-A) password: r2zi

The COCO pretain weight is here:
  -  [COCO Weight](https://pan.baidu.com/s/1n_9pSC2kZiuMiCq7EMQdRQ) password: tkze

## Dataset preparation
Create a TXT document with each line containing the address, bbox and label of each image.

## Train
### Data pre-process
Firstly, resize the image to (416, 416).
### Train model
To train models, run following commands.
```
$ python train.py
```
The pretrained model on the COCO is loaded. 
We first freeze the parameters of backbone to train 25 epochs, and then unfreeze the parameters of backbone for the training of another 25 epochs.

Note that all hyper parameters are set done in the train.py.
If you want to change, JUST DO IT!!

## Test
Run following commands to generate the submission.json file of the testing results.
```
$ python test.py
```

## Speed
The speed of YOLOv4 I test on the [Google Colab](https://colab.research.google.com) is as follows.
![here](https://raw.githubusercontent.com/cocozxh/CS-IOC5008/main/HW2/speed1.png)
