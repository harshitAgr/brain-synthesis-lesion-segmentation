import tensorflow as tf
import numpy as np
import pickle, random
from scipy import ndimage

IMG_WIDTH = 256
IMG_HEIGHT = 256

def _decode_pkl(filename):
  with open(filename, 'rb') as f:
    imageA = pickle.load(f)
  return imageA

def load_image(image_file_A, image_file_B, is_train):
  imageA = _decode_pkl(image_file_A)
  imageA = tf.cast(imageA, tf.float32)

  imageB = _decode_pkl(image_file_B)
  imageB = tf.cast(imageB, tf.float32)

  if len(imageA.shape) < 4:
    imageA = tf.expand_dims(imageA, -1)
  if len(imageB.shape) < 4:
    imageB = tf.expand_dims(imageB, -1)

  if is_train:
    # random jittering

    SCALE = 1.18
    ORDER = 3
    # it takes too long to perform random cropping in 4D...
    #imageA = ndimage.interpolation.zoom(imageA, [1, SCALE, SCALE, 1], order = ORDER)
    #imageB = ndimage.interpolation.zoom(imageB, [1, SCALE, SCALE, 1], order = ORDER)

    # # randomly cropping to 256 x 256 x 3
    # stacked_image = tf.stack([imageA, imageB], axis=0)
    # cropped_image = tf.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    # imageA, imageB = cropped_image[0], cropped_image[1]

    #idx = random.randrange(286-256)
    #imageA = imageA[:, idx:idx+256, idx:idx+256, :]
    #imageB = imageB[:, idx:idx+256, idx:idx+256, :]

    if np.random.random() > 0.5:
      # random mirroring
      #imageA = tf.image.flip_left_right(imageA)
      #imageB = tf.image.flip_left_right(imageB)
      imageA = np.flip(imageA, 2)
      imageB = np.flip(imageB, 2)

  else:
    pass

  # normalizing the images to [-1, 1]
  imageA = (imageA / 127.5) - 1
  imageB = (imageB / 127.5) - 1

  return imageA, imageB
