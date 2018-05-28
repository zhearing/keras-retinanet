#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import os
import numpy as np
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def retinanet(model, image_path, threshold):
    #print(model.summary())
    image = read_image_bgr(image_path)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    #start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #print("processing time: ", time.time() - start)

    # correct for image scale
    boxes //= scale

    a = []
    b = []
    c = []
    
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < threshold:
            break
        box = box.astype(int)
        score = score.astype(float)
        label = label.astype(int)
        a.append(box)
        b.append(score)
        c.append(label)
    return a, b, c














