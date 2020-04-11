import tensorflow as tf
import os
import random
import sys
import pathlib
from PIL import Image

# 测试集数量
_NUM_TEST = 2000
# 随机种子
_RANDOM_SEED = 0
# 定义数据块数量
_NUM_SHARDS = 5
# 数据集路径
DATASET_DIR = "/root/quan/fruits-360/Training/"
# 标签文件存放名字
#LABEL_FILENAME = "labels.txt"
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
label_name = sorted(item.name for item in test_path.glob('*/')
                    if item.is_dir())


# 定义tfrecord文件的路径+文字
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'image-%s.tfrecords-%05d-of-%05d' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


# 判断tfrecord 文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            # 定义tfrecord文件的路径+名字
            output_filename = _get_dataset_filename(dataset_dir, split_name,
                                                    shard_id)
        if not tf.io.gfile.Exists(output_filename):
            return False
    return True


# 获取所有文件以及分类
def _get_filenames_and_classes(dataset_dir):
    # 数据目录
    directories = []
    # 分类名称
    class_names = []
    for filename in os.listdir(dataset_dir):
        # 合并文件路径
        path = os.path.join(dataset_dir, filename)
        # 判断该路径是否为目录
        if os.path.isdir(path):
            # 加入数据目录
            directories.append(path)
            class_names.append(filename)
    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
    return photo_filenames, class_names


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, class_id):
    return tf.train.Example(features=tf.train.Features(
        feature={
            'image': bytes_feature(image_data),
            'label': int64_feature(class_id)
        }))


def write_label_file(labels_to_class_names, dataset_dir, filename=label_name):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write("%d:%s\n" % (label, class_name))


def _convert_dataset(split_name, file_names, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'test']
    # 计算每个数据块有多少数据
    num_per_shard = int(len(file_names) / _NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session():
            for shard_id in range(_NUM_SHARDS):
                # 定义tfrecord文件的路径+名字
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(
                        output_filename) as tfrecord_writer:
                    # 每一个数据块的开始的位置
                    start_ndx = shard_id * num_per_shard
                    # 每一个数据块的最后的位置
                    end_ndx = min((shard_id + 1) * num_per_shard,
                                  len(file_names))
                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write(
                                "\r>>[%s] Converting image %d/%d shard %d" %
                                (split_name, i + 1, len(file_names), shard_id))
                            sys.stdout.flush()
                            # 读取图片
                            image_data = tf.gfile.FastGFile(
                                file_names[i], 'rb').read()
                            # 获得图片类别名称
                            class_name = os.path.basename(
                                os.path.dirname(file_names[i]))
                            # 找到类别名称对应的id
                            class_id = class_names_to_ids[class_name]
                            example = image_to_tfexample(image_data, class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read:", file_names[i])
                            print("Error", e)
                            print("Skip~\n")
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    if _dataset_exists(DATASET_DIR):
        print("tfrecord已存在")
    else:
        # 获取图片和分类
        photo_filenames, class_names = _get_filenames_and_classes(DATASET_DIR)
        print(photo_filenames, class_names)
        # 把分类转为字典格式，类似于（'a':0,'b':1,'c':2....）
        class_name_to_ids = dict(zip(class_names, range(len(class_names))))

        # 把数据切分为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)  # 打乱数组
        traning_filenames = photo_filenames[_NUM_TEST:]  # 500~ 为训练集
        testing_filenames = photo_filenames[:_NUM_TEST]  # 0~500 为测试集

        # 数据转换
        _convert_dataset('train', traning_filenames, class_name_to_ids,
                         DATASET_DIR)
        _convert_dataset('test', testing_filenames, class_name_to_ids,
                         DATASET_DIR)

        # 输出labels
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, DATASET_DIR)
