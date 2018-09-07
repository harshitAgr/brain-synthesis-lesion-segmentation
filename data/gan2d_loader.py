import tensorflow as tf
import glob2 as glob
import os, random
import numpy as np


def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=1)
  image_resized = tf.image.resize_images(image_decoded, [64, 64])
  return image_resized

def _train_preprocess(image):
  image = (image - 127.5) / 127.5
  return image

def data_loader(data_dir):
  files = glob.glob(os.path.join(data_dir, '*.png'))

  random.shuffle(files)

  dataset = tf.data.Dataset.from_tensor_slices((files))
  dataset = dataset.shuffle(10000)
  dataset = dataset.map(_parse_function, num_parallel_calls=16)
  dataset = dataset.map(_train_preprocess, num_parallel_calls=16)
  return dataset

