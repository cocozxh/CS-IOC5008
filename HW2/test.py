import time
import numpy as np
from yolo import YOLO
from PIL import Image
import os

#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
def detect_image(image_path):
    print('Start detect!')
    yolo = YOLO()
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
        pass
    else:
        r_image = yolo.detect_image(image)
        r_image.save(image_path.split('.')[0] + '_result.png')
    print('Finish detect!')

detect_image('pre_test/3.png')    

#-------------------------------------#
#       对文件夹图片进行预测
#-------------------------------------#
def detect_images(image_path):
    print('Start detect!')
    predictions = []
    yolo = YOLO()
    path_test = '/data/zihaosh/data_hw2/'
    num = len(os.listdir(path_test))
    for i in range(1, num+1):
        try:
            image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
            pass
        else:
            r_image = yolo.detect_image(image)
            r_image.save(image_path.split('.')[0] + '_result.png')
        print('Finish detect!')

predictions = []
img_dic = {'l':[1,2,3],'k':[4,5,6]}        
import json        
with open('/output/submission.json','w') as file_obj:
    json.dump(img_dic,file_obj)    

