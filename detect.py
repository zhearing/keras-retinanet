#! /usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import time
import csv
import os
from PIL import Image
from keras_retinanet import models
from retinanet import retinanet
from retinanet import get_session

save_dir    = '/media/zyzhong/Data/data3/HualuCUP/result/'
test_dir    = '/media/zyzhong/Data/data3/HualuCUP/ehualu/test_a/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

keras.backend.tensorflow_backend.set_session(get_session())
model_path = os.path.join('model_data', 'resnet50_coco_best_v2.1.0.h5')
model = models.load_model(model_path, backbone_name='resnet50')
# if the model is not converted to an inference model, use the line below
#model = models.load_model(model_path, backbone_name='resnet50', convert_model=True)
img_list = os.listdir(test_dir)
img_list = sorted(img_list)

save_csv_path = os.path.join(save_dir, 'result.csv')
csvfile = open(save_csv_path, 'w')
writer = csv.writer(csvfile)
writer.writerow(['name', 'coordinate'])
x = {}
y = {}
w = {}
h = {}

def pred_clear():
    x = {}
    y = {}
    w = {}
    h = {}
    return

total = len(img_list)
print(('testing images = %d' % total))
remain = total
start_time = time.time()

for i, img_name in enumerate(img_list):
    img_path = os.path.join(test_dir, img_name)
    boxes, score, predicted_class = retinanet(model, img_path, 0.3)

    number = len(predicted_class)
    coordinate = ''
    for i in range(number):
        x[i] = boxes[i][0]
        y[i] = boxes[i][1]
        w[i] = boxes[i][2] - boxes[i][0]
        h[i] = boxes[i][3] - boxes[i][1]   
        if predicted_class[i] == 2 or 5 or 7:
            coordinate = coordinate + str(x[i]) + '_' + str(y[i]) + '_'+ str(w[i]) + '_' + str(h[i]) + ';'
            #print(score[i])        
    writer.writerow([img_name, coordinate[:-1]])
    pred_clear()

    remain -= 1
    if remain % 100 == 0:
        print(('Remain: %d  \t Time Taken: %.2f min' % (remain, (time.time() - start_time) / 60.0)))

csvfile.close()


