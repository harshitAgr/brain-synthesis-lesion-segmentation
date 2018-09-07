import tensorflow as tf
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256

def load_image(image_file_A, image_file_B, is_train):
  imageA = tf.read_file(image_file_A)
  imageB = tf.read_file(image_file_B)
  imageA = tf.image.decode_png(imageA, channels=3)
  imageB = tf.image.decode_png(imageB, channels=3)

  imageA = tf.cast(imageA, tf.float32)
  imageB = tf.cast(imageB, tf.float32)

  if is_train:
    # random jittering

    # resizing to 286 x 286 x 3
    # method = 2 indicates using "ResizeMethod.NEAREST_NEIGHBOR"
    imageA = tf.image.resize_images(imageA, [300, 300],
                                         align_corners=True, method=2)
    imageB = tf.image.resize_images(imageB, [300, 300],
                                        align_corners=True, method=2)

    # randomly cropping to 256 x 256 x 3
    stacked_image = tf.stack([imageA, imageB], axis=0)
    cropped_image = tf.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    imageA, imageB = cropped_image[0], cropped_image[1]

    if np.random.random() > 0.5:
      # random mirroring
      imageA = tf.image.flip_left_right(imageA)
      imageB = tf.image.flip_left_right(imageB)

  else:
    imageA = tf.image.resize_images(imageA, size=[IMG_HEIGHT, IMG_WIDTH],
                                         align_corners=True, method=2)
    imageB = tf.image.resize_images(imageB, size=[IMG_HEIGHT, IMG_WIDTH],
                                        align_corners=True, method=2)

  # normalizing the images to [-1, 1]
  imageA = (imageA / 127.5) - 1
  imageB = (imageB / 127.5) - 1

  return imageA, imageB
