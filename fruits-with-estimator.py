#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import models

gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True) # 其实gpus本身就只有一个元素
os.environ['CUDA_VISIBLE_DEVICES']='1' 
# In[3]:

data_root_orig = "/root/quan/fruits-360/Training/"
data_root = pathlib.Path(data_root_orig)
print(data_root)

# In[78]:

test_data_path = "/root/quan/fruits-360/Test/"
test_path = pathlib.Path(test_data_path)
print(test_path)
# In[5]:

test_image_path = list(test_path.glob('*/*'))
test_image_path = [str(path) for path in test_image_path]

# In[21]:

traing_image_path = list(data_root.glob('*/*'))
traing_image_path = [str(path) for path in traing_image_path]

# In[7]:

label_name = sorted(item.name for item in test_path.glob('*/')
                    if item.is_dir())

# In[8]:

traing_image_count = len(traing_image_path)
test_image_count = len(test_image_path)
traing_image_count, test_image_count

# In[55]:


def decode_img(img_raw):
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    print(img_tensor.shape)
    #img_tensor = (img_tensor/127.5) - 1
    tf_fianl = tf.image.resize(img_tensor, [160, 160])

    # 格式化0-1
    img = tf.image.convert_image_dtype(tf_fianl, tf.float32)
    img = (img / 127.5) - 1
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == label_name


# In[48]:

#list_ds = tf.data.Dataset.from_tensor_slices(traing_image_path)
#labeled_ds = list_ds.map(process_path,
#                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#labeled_ds.shuffle(buffer_size=100)

# In[56]:


def input_fn(batch_size):
    list_ds = tf.data.Dataset.from_tensor_slices(traing_image_path)
    labeled_ds = list_ds.map(process_path,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return labeled_ds.shuffle(buffer_size=100).batch(batch_size)

def train_input_fn(batch_size):
    list_ds = tf.data.Dataset.from_tensor_slices(test_image_path)
    labeled_ds = list_ds.map(process_path,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return labeled_ds.shuffle(buffer_size=100).batch(batch_size)

# In[60]:

# model = models.Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1)
# ])
#keras_mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224,
#                                                                    3),
#                                                       include_top=False)
#keras_mobilenet_v2.trainable = False

#estimator_model = tf.keras.Sequential([
#    keras_mobilenet_v2,
#    tf.keras.layers.GlobalAveragePooling2D(),
#    tf.keras.layers.Dense(1)
#])

estimator_model = models.Sequential([
    Conv2D(16,
           3,
           padding='same',
           activation='softmax',
           input_shape=(160, 160, 3)),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
# Compile the model
estimator_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])


model_dir = "/tmp/fruits/"
est_model = tf.keras.estimator.model_to_estimator(keras_model=estimator_model, model_dir=model_dir)

# In[62]:

est_model.train(input_fn=lambda: input_fn(40), steps=100)
est_model.evaluate(input_fn=lambda: input_fn(40), steps=100)
