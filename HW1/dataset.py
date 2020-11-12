import os
import shutil
import pandas as pd
import time

time_start = time.time()

path = 'data/'
path_images = path + 'training_data/training_data/'
train_save_path = path + 'train/'
path_labels = path + 'training_labels.csv'

# 读取所有图片的文件名
images = os.listdir(path_images)

# 读取所有图片的id以及类别
kinds = []
id_label = []
raw_data = pd.read_csv(path_labels)
for i in range(raw_data.shape[0]):
    kinds.append(raw_data.iloc[i][1])
    id_label.append(raw_data.iloc[i].values)

# 划分
num = len(images)  # 图像的总个数
for k in range(num):
    file_name = images[k]
    id = int(file_name.split('.')[0])
    label = 'a'
    for i in range(len(id_label)):  # 找到类别
        if id_label[i][0] == id:
            print('yes')
            label = id_label[i][1]
            del id_label[i]
            print('the length is', len(id_label))
            break

    if os.path.exists(train_save_path + label):
        shutil.copy(path_images + file_name,
                    train_save_path+label+'/'+file_name)
    else:
        os.makedirs(train_save_path+label)
        shutil.copy(path_images + file_name,
                    train_save_path+label+'/'+file_name)
    print('%s处理完毕!' % file_name)

time_end = time.time()
print('完毕, 耗时%s!!' % (time_end - time_start))
