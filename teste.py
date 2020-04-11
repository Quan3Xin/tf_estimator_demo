#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import models
gpus = tf.config.list_physical_devices('GPU')  # tf2.1版本该函数不再是experimental
print(gpus)  # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True)  # 其实gpus本身就只有一个元素
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2072)])
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# In[3]:

# In[2]:

# the number of classes of images

# 组装 label名称和标签名称 例如 {“name”：1}
# In[7]:

# def input_fn(filenames, training):
#     dataset = tf.data.TFRecordDataset(filenames)
#     dataset = dataset.map(parse_data)

#     if training:
#         dataset = dataset.shuffle(buffer_size=50000)
#     dataset = dataset.batch(FLAGS.batch_size)
#     if training:
#         dataset = dataset.repeat()


#     iterator = dataset.make_one_shot_iterator()
#     features, labels = iterator.get_next()
#     return features, labels
def input_fn(record_list):
    list_ds = tf.data.TFRecordDataset(filenames=record_list)

    dataset = list_ds.map(_parse_image_function)
    data = dataset.map(decode_img)

    return data


def decode_img(data):
    #list_ds = tf.data.TFRecordDataset(filenames = record_list)

    #data = list_ds.map(_parse_image_function)
    img_raw = data["image_raw"]
    #height = data["height"]

    #wight = data["width"]
    label = data["label"]

    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    #imgs = tf.image.resize(img_tensor, [10,10])
    img = tf.image.convert_image_dtype(img_tensor, tf.float32)

    img = tf.image.resize(img, [400, 400])
    print(img)
    return img, label


# In[10]:
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}
"""

image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}
"""


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def process_data(raw_image_dataset):

    dataset = raw_image_dataset.map(_parse_image_function)
    # In[15]:

    train_data = dataset.map(decode_img).shuffle(10).batch(
        batch_size=40).repeat()
    return train_data


# In[ ]:

# In[27]:


# 提供组装 数据方法
def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def format_img(data):

    img_raw = data["image_raw"]
    height = data["height"]

    wight = data["width"]
    label = data["label"]

    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.convert_image_dtype(img_tensor, tf.float32)

    img = tf.image.resize(img, [10, 10])
    return img, label


def train_net(train_data):
    model_new = models.Sequential([
        Conv2D(16,
               3,
               padding='same',
               activation='softmax',
               data_format="channels_last",
               input_shape=(400, 400, 3)),
        MaxPooling2D(),
        Dropout(0.6),
        Conv2D(32, 3, padding='same', activation='softmax'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])
    model_new.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model_new.summary()

    # Compile the model
    # estimator_model.compile(
    #     optimizer='adam',
    #     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #     metrics=['accuracy'])
    # model_dir = "/tmp/note-fruits-tfrecord/"
    # est_model = tf.keras.estimator.model_to_estimator(keras_model=estimator_model)

    # In[ ]:

    model_new.fit(train_data, epochs=1000, steps_per_epoch=1000)


if __name__ == "__main__":
    print("------- start --------------")
    data_root_orig = "/root/quan/fruits-360/Training/"
    data_root = pathlib.Path(data_root_orig)
    label_name = sorted(item.name for item in data_root.glob('*/')
                        if item.is_dir())
    values = [i for i in list(range(len(label_name)))]
    classes = dict(zip(label_name, values))

    # In[9]:

    os.chdir("/root/quan/fruits-360/tfrecord/")
    name_list = [k for _, _, k in os.walk(".")]
    name_list = name_list[0]
    raw_image_dataset = tf.data.TFRecordDataset(filenames=[name_list])
    train_data = process_data(raw_image_dataset)
    train_net(train_data)
