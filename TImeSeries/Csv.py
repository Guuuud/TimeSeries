


import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# URl = "/Users/lee/Desktop/Project/qinghua.csv"
train_file_path = tf.keras.utils.get_file("qinghua.csv","/Users/lee/Desktop/Project/qinghua.csv")
# 让 numpy 数据更易读。
np.set_printoptions(precision=3, suppress=True)
# print(train_file_path)
