import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from utils.utils import *
import json
import time


class YOLO(object):
    _defaults = {
        "model_path": '/data/zihaosh/hw2_load/final.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/mask_classes.txt',
        "model_image_size": (608, 608, 3),
        "confidence": 0.01,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    def generate(self):

        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        print('Loading pretrained weights.')

        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(self.model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('Finish loading!')

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=0.3)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image, [(1, 1, 1, 1)], [1], [1]

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin = np.expand_dims(top_bboxes[:, 0], -1)
        top_ymin = np.expand_dims(top_bboxes[:, 1], -1)
        top_xmax = np.expand_dims(top_bboxes[:, 2], -1)
        top_ymax = np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=10)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = max(0, np.floor(top + 0.5).astype('int64'))
            left = max(0, np.floor(left + 0.5).astype('int64'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int64'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int64'))

            # 画框框
            label = '{}: {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(2):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image, boxes, top_conf, top_label


# -------------------------------------#
#       对文件夹图片进行预测
# -------------------------------------#
def detect_images():
    print('Start detect!')
    predictions = []
    yolo = YOLO()
    path_test = '/data/zihaosh/data_hw2/test/'
    num = len(os.listdir(path_test))
    for i in range(1, num + 1):
        image_path = path_test + str(i) + '.png'
        print(image_path)
        img_dic = {}
        try:
            image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
            pass
        else:
            r_image, boxes, top_conf, top_label = yolo.detect_image(image)

            bbox = []
            for j in range(len(boxes)):
                top, left, bottom, right = boxes[j]
                top = max(0, np.floor(top + 0.5).astype('int64'))
                left = max(0, np.floor(left + 0.5).astype('int64'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int64'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int64'))
                bbox.append((top, left, bottom, right))
            img_dic['bbox'] = bbox
            img_dic['score'] = list(top_conf)
            img_dic['label'] = list(top_label)
            if not os.path.exists('/output/out4/'):
                os.makedirs('/output/out4/')
            r_image.save('/output/out4/' + str(i) + '_result.png')
        predictions.append(img_dic)
        print('Finish detect!', i)
    return predictions


time_start = time.time()
predictions = detect_images()
print(len(predictions))
for i in range(len(predictions)):
    if i % 3000 == 0:
        print(i)
    predictions[i]['label'] = [10 if j == 0 else j for j in predictions[i]['label']]
time_end = time.time()
print('total time', time_end - time_start)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


for i in range(len(predictions)):
    predictions[i]['label'] = [10 if j == 0 else j for j in predictions[i]['label']]

with open('/output/2040.json', 'w') as file_obj:
    json.dump(predictions, file_obj, cls=NpEncoder)
print(len(predictions))
