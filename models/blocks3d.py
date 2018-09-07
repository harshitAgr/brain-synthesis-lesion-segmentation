from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

layers = tf.keras.layers


data_format = 'channels_last'


class _IdentityBlock(tf.keras.Model):
  """_IdentityBlock is the block that has no conv layer at shortcut.
  Args:
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    data_format: data_format for the input ('channels_first' or
      'channels_last').
  """

  def __init__(self, kernel_size, filters, stage, block, data_format):
    super(_IdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv3D(
        filters1, (1, 1, 1), name=conv_name_base + '2a', data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv3D(
        filters2,
        kernel_size,
        padding='same',
        data_format=data_format,
        name=conv_name_base + '2b')
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv3D(
        filters3, (1, 1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
  """_ConvBlock is the block that has a conv layer at shortcut.
  Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
      strides: strides for the convolution. Note that from stage 3, the first
       conv layer at main path is with strides=(2,2), and the shortcut should
       have strides=(2,2) as well.
  """

  def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               data_format,
               strides=(2, 2, 2)):
    super(_ConvBlock, self).__init__(name='')
    
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv3D(
        filters1, (1, 1, 1),
        strides=strides,
        name=conv_name_base + '2a',
        data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv3D(
        filters2,
        kernel_size,
        padding='same',
        name=conv_name_base + '2b',
        data_format=data_format)
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv3D(
        filters3, (1, 1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

    self.conv_shortcut = layers.Conv3D(
        filters3, (1, 1, 1),
        strides=strides,
        name=conv_name_base + '1',
        data_format=data_format)
    self.bn_shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    shortcut = self.conv_shortcut(input_tensor)
    shortcut = self.bn_shortcut(shortcut, training=training)

    x += shortcut
    return tf.nn.relu(x)


class _ConvTBlock(tf.keras.Model):
  """_ConvBlock is the block that has a conv layer at shortcut.
  Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
      strides: strides for the convolution. Note that from stage 3, the first
       conv layer at main path is with strides=(2,2), and the shortcut should
       have strides=(2,2) as well.
  """

  def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               data_format,
               strides=(2, 2, 2)):
    super(_ConvTBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv3DTranspose(
        filters1, (1, 1, 1),
        strides=strides,
        name=conv_name_base + '2a',
        data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv3DTranspose(
        filters2,
        kernel_size,
        padding='same',
        name=conv_name_base + '2b',
        data_format=data_format)
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv3DTranspose(
        filters3, (1, 1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

    self.conv_shortcut = layers.Conv3DTranspose(
        filters3, (1, 1, 1),
        strides=strides,
        name=conv_name_base + '1',
        data_format=data_format)
    self.bn_shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    shortcut = self.conv_shortcut(input_tensor)
    shortcut = self.bn_shortcut(shortcut, training=training)

    x += shortcut
    return tf.nn.relu(x)


def conv_block(filters, stage, block, strides=(2, 2, 2)):
  return _ConvBlock(
      3,
      filters,
      stage=stage,
      block=block,
      data_format=data_format,
      strides=strides)

def id_block(filters, stage, block):
  return _IdentityBlock(
      3, filters, stage=stage, block=block, data_format=data_format)
