
import os
import time
import random

time_start = time.time()

# 文件路径
path = 'data/'
path_train = path + 'train/'
path_val = path + 'validation/'

# 读取所有图片的文件名
labels = os.listdir(path_train)
if '.DS_Store' in labels:
    labels.remove('.DS_Store')

p = 0.1

# 划分
num = len(labels)  # 标签的总个数
for k in range(num):
    file_name = labels[k]
    path_label = path_train + file_name
    name_label = os.listdir(path_label)
    if '.DS_Store' in name_label:
        name_label.remove('.DS_Store')
    num_val = int(len(name_label) * p)
    name_label_val = random.sample(name_label, num_val)

    for name in name_label_val:
        if os.path.exists(path_val + file_name):
            os.rename(path_train+file_name+'/'+name, path_val+file_name+'/'+name)
        else:
            os.makedirs(path_val + file_name)
            os.rename(path_train+file_name+'/'+name, path_val+file_name+'/'+name)
    print('%s处理完毕!' % file_name)

time_end = time.time()
print('完毕, 耗时%s!!' % (time_end - time_start))

