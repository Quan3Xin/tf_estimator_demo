#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os

# In[50]:
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SeparableConv2D
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[3]:

# In[4]:


def show_image(image):
    plt.figure()
    plt.imshow(train_images[100])
    plt.colorbar()
    plt.grid(False)
    plt.show()


# In[155]:


def get_predict_data(data):
    # predict_data = train_images[100]

    pre_list = [
        str(i) for i in list(pathlib.Path("/data/birds/valid/").glob("*/*"))
    ]
    predict_data = list()
    for a in pre_list[::100]:
        print(a)
        a = tf.io.read_file(a).numpy()

        d = tf.image.resize(tf.image.decode_jpeg(a), [150, 150])
        d = d.numpy()

        d = d.reshape(1, 150, 150, 3).tolist()
        predict_data.append(d)

    for i in predict_data:
        pre_data = model.predict(predict_data[0])
        print(pre_data)
        print(np.argmax(pre_data))


# In[55]:


# In[18]:
def data_path():
    # taring_data

    taring_data_path = "/data/birds/train/*/*"
    taring_data_ = pathlib.Path(taring_data_path)
    # test data
    test_data_path = "/data/birds/test/*/*"
    test_data_ = pathlib.Path(test_data_path)

    # vaildata
    valid_data_path = "/data/birds/valid/*/*"
    valid_data_ = pathlib.Path(valid_data_path)

    # label_name = sorted(item.name for item in test_data_.glob('*/')
    #                     if item.is_dir())
    return taring_data_, test_data_, valid_data_


# In[56]:

# In[5]:


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [150, 150])
    return image, label


#par_ds  = tf.data.Dataset.from_tensor_slices(traing_image_path).map(parse_image).shuffle(buffer_size=2000).batch(batch_size=100).repeat()


# In[20]:
def process_model():

    taring_data_, valid_data_, test_data_ = data_path()
    traing_ds = tf.data.Dataset.list_files(
        str(taring_data_)).map(parse_image).shuffle(buffer_size=2000).batch(
            batch_size=100).repeat()

    validation_ds = tf.data.Dataset.list_files(
        str(valid_data_)).map(parse_image).shuffle(buffer_size=2000).batch(
            batch_size=100).repeat()
    test_ds = tf.data.Dataset.list_files(
        str(test_data_)).map(parse_image).shuffle(buffer_size=2000).batch(
            batch_size=100).repeat()

    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model_dir = "../estir/model/"

    history = model.fit(traing_ds,
                        epochs=10,
                        steps_per_epoch=100,
                        validation_data=validation_ds,
                        validation_steps=5)
    tf.saved_model.save(model, model_dir)


# In[47]:
def evaluate_model(model):
    model.evaluate(test_ds, steps=100)


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices("GPU")[0]
    tf.config.experimental.set_memory_growth(gpus, True)
    process_model()
