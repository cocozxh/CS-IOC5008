# Super resolution

Code for homework 4 in the Selected Topics in Visual Recognition using Deep Learning.
I choose the VDSR as my network.
## Catalog
- [Instance segmentation](#super-resolution)
  - [Catalog](#Catalog)
  - [Installation](#Installation)
    - [Dataset](#Dataset)
    - [Requirements](#Requirements)
    - [Pretrained model](#Pretrained-model)
  - [Train](#train)
    - [Train model](#train-model)
  - [Test](#test)

## Installation
### Dataset
The dataset for this homework is here:
  - [Link](https://drive.google.com/drive/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x)
### Requirements
- Python >= 3.6
- PyTorch >= 1.3.0
### Pretrained model
The final model I traiined is here:
  -  [Final Model](https://pan.baidu.com/s/184g9QWYgCMAeid_zHmdB4g) extraction code: orwy


## Train
### Train model
To train models, run following commands.
```
$ python main.py --dataset train_hr_hw4 --cuda --upscale_factor 3 --crop_size 256 --batch_size 16 --test_batch_size 16 --epochs 100 --step 20
```

Note that all hyper parameters are set done in the train.py.

## Test
Run following commands to generate the submission.json file of the testing results.
```
$ python run2.py --cuda --scale_factor 3 --model model_epoch_100.pth --input_image /data/zihaosh/hw4/testing_lr_images/
```
