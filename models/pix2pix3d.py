import tensorflow as tf
from models import blocks3d
import numpy as np

layers = tf.keras.layers

data_format = 'channels_last'

OUTPUT_CHANNELS = 3
LAMBDA = 100


class Downsample(tf.keras.Model):

  def __init__(self, filters, size, strides=2, apply_batchnorm=True):
    super(Downsample, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    initializer = tf.random_normal_initializer(0., 0.02)

    self.conv1 = tf.keras.layers.Conv3D(filters,
                                        (size, size, size),
                                        strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)
    if self.apply_batchnorm:
        self.batchnorm = tf.keras.layers.BatchNormalization()

  def call(self, x, training):
    x = self.conv1(x)
    if self.apply_batchnorm:
        x = self.batchnorm(x, training=training)
    x = tf.nn.leaky_relu(x)
    return x


class Upsample(tf.keras.Model):

  def __init__(self, filters, size, strides=2, apply_dropout=False):
    super(Upsample, self).__init__()
    self.apply_dropout = apply_dropout
    initializer = tf.random_normal_initializer(0., 0.02)

    self.up_conv = tf.keras.layers.Conv3DTranspose(filters,
                                                   (size, size, size),
                                                   strides=strides,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   use_bias=False)
    self.batchnorm = tf.keras.layers.BatchNormalization()
    if self.apply_dropout:
        self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, x1, x2, training):
    x = self.up_conv(x1)
    x = self.batchnorm(x, training=training)
    if self.apply_dropout:
        x = self.dropout(x, training=training)
    x = tf.nn.relu(x)
    x = tf.concat([x, x2], axis=-1)
    return x


class Generator(tf.keras.Model):

  def __init__(self):
    super(Generator, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)

    self.down1 = Downsample(64, 4, apply_batchnorm=False)
    self.down2 = Downsample(128, 4)
    self.down3 = Downsample(256, 4)
    self.down4 = Downsample(512, 4, [1, 2, 2])
    self.down5 = Downsample(512, 4, [1, 2, 2])
    self.down6 = Downsample(512, 4, [1, 2, 2])
    self.down7 = Downsample(512, 4, [1, 2, 2])
    self.down8 = Downsample(512, 4, [3, 2, 2])
    
    self.up1 = Upsample(512, 4, [3, 2, 2], apply_dropout=True)
    self.up2 = Upsample(512, 4, [1, 2, 2], apply_dropout=True)
    self.up3 = Upsample(512, 4, [1, 2, 2], apply_dropout=True)
    self.up4 = Upsample(512, 4, [1, 2, 2])
    self.up5 = Upsample(256, 4, [1, 2, 2])
    self.up6 = Upsample(128, 4)
    self.up7 = Upsample(64, 4)

    self.last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS,
                                                (4, 4, 4),
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer)

  @tf.contrib.eager.defun
  def call(self, x, training):
    # x shape == (bs, 256, 256, 3)
    x1 = self.down1(x, training=training) # (bs, 128, 128, 64)
    x2 = self.down2(x1, training=training) # (bs, 64, 64, 128)
    x3 = self.down3(x2, training=training) # (bs, 32, 32, 256)
    x4 = self.down4(x3, training=training) # (bs, 16, 16, 512)
    x5 = self.down5(x4, training=training) # (bs, 8, 8, 512)
    x6 = self.down6(x5, training=training) # (bs, 4, 4, 512)
    x7 = self.down7(x6, training=training) # (bs, 2, 2, 512)
    x8 = self.down8(x7, training=training) # (bs, 1, 1, 512)
     
    x9 = self.up1(x8, x7, training=training) # (bs, 2, 2, 1024)
    x10 = self.up2(x9, x6, training=training) # (bs, 4, 4, 1024)
    x11 = self.up3(x10, x5, training=training) # (bs, 8, 8, 1024)
    x12 = self.up4(x11, x4, training=training) # (bs, 16, 16, 1024)
    x13 = self.up5(x12, x3, training=training) # (bs, 32, 32, 512)
    x14 = self.up6(x13, x2, training=training) # (bs, 64, 64, 256)
    x15 = self.up7(x14, x1, training=training) # (bs, 128, 128, 128)

    x16 = self.last(x15) # (bs, 256, 256, 3)
    x16 = tf.nn.tanh(x16)

    return x16


class DiscDownsample(tf.keras.Model):

  def __init__(self, filters, size, apply_batchnorm=True):
    super(DiscDownsample, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    initializer = tf.random_normal_initializer(0., 0.02)

    self.conv1 = tf.keras.layers.Conv3D(filters,
                                        (size, size, size),
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)
    if self.apply_batchnorm:
        self.batchnorm = tf.keras.layers.BatchNormalization()

  def call(self, x, training):
    x = self.conv1(x)
    if self.apply_batchnorm:
        x = self.batchnorm(x, training=training)
    x = tf.nn.leaky_relu(x)
    return x

class Discriminator(tf.keras.Model):

  def __init__(self):
    super(Discriminator, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)

    self.down1 = DiscDownsample(64, 4, False)
    self.down2 = DiscDownsample(128, 4)
    self.down3 = DiscDownsample(256, 4)

    # we are zero padding here with 1 because we need our shape to
    # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
    self.zero_pad1 = tf.keras.layers.ZeroPadding3D()
    self.conv = tf.keras.layers.Conv3D(512,
                                       (4, 4, 4),
                                       strides=1,
                                       kernel_initializer=initializer,
                                       use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()

    # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
    self.zero_pad2 = tf.keras.layers.ZeroPadding3D()
    self.last = tf.keras.layers.Conv3D(1,
                                       (4, 4, 4),
                                       strides=1,
                                       kernel_initializer=initializer)

  @tf.contrib.eager.defun
  def call(self, inp, tar, training):
    # concatenating the input and the target
    x = tf.concat([inp, tar], axis=-1) # (bs, 256, 256, channels*2)
    x = self.down1(x, training=training) # (bs, 128, 128, 64)
    x = self.down2(x, training=training) # (bs, 64, 64, 128)
    x = self.down3(x, training=training) # (bs, 32, 32, 256)

    x = self.zero_pad1(x) # (bs, 34, 34, 256)
    x = self.conv(x)      # (bs, 31, 31, 512)
    x = self.batchnorm1(x, training=training)
    x = tf.nn.leaky_relu(x)

    x = self.zero_pad2(x) # (bs, 33, 33, 512)
    # don't add a sigmoid activation here since
    # the loss function expects raw logits.
    x = self.last(x)      # (bs, 30, 30, 1)

    return x


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_real_output),
                                              logits = disc_real_output)
  generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_generated_output),
                                                   logits = disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss



def  generator_loss (disc_generated_output, gen_output, target):
  gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output),
                                             logits = disc_generated_output)
  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss


class GeneratorRes(tf.keras.Model):

  def __init__(self):
    super(GeneratorRes, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)

    bn_axis = 3

    def conv_block(filters, stage, block, strides=(2, 2, 2)):
      return blocks3d._ConvBlock(
          3,
          filters,
          stage=stage,
          block=block,
          data_format=data_format,
          strides=strides)

    def convt_block(filters, stage, block, strides=(2, 2, 2)):
        return blocks3d._ConvTBlock(
            3,
            filters,
            stage=stage,
            block=block,
            data_format=data_format,
            strides=strides)

    def id_block(filters, stage, block):
      return blocks3d._IdentityBlock(
          3, filters, stage=stage, block=block, data_format=data_format)

    self.conv1 = layers.Conv3D(
        64, (7, 7, 7),
        strides=(1, 1, 1),
        data_format=data_format,
        padding='same',
        name='conv1')
    bn_axis = 1 if data_format == 'channels_first' else 3
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
    self.max_pool = layers.MaxPooling3D(
        (3, 3, 3), strides=(2, 2, 2), data_format=data_format)

    self.l2a = conv_block([64, 64, 64], stage=2, block='a')
    self.l2b = id_block([64, 64, 64], stage=2, block='b')
    self.l2c = id_block([64, 64, 64], stage=2, block='c')

    self.l3a = conv_block([128, 128, 512], stage=3, block='a')
    self.l3b = id_block([128, 128, 512], stage=3, block='b')
    self.l3c = id_block([128, 128, 512], stage=3, block='c')
    self.l3d = id_block([128, 128, 512], stage=3, block='d')

    self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
    self.l4b = id_block([256, 256, 1024], stage=4, block='b')
    self.l4c = id_block([256, 256, 1024], stage=4, block='c')
    self.l4d = id_block([256, 256, 1024], stage=4, block='d')
    self.l4e = id_block([256, 256, 1024], stage=4, block='e')
    self.l4f = id_block([256, 256, 1024], stage=4, block='f')

    self.l5a = convt_block([256, 256, 512], stage=5, block='a')
    self.l5b = id_block([256, 256, 512], stage=5, block='b')
    self.l5c = id_block([256, 256, 512], stage=5, block='c')
    self.l5d = id_block([256, 256, 512], stage=5, block='c')

    self.l6a = convt_block([128, 128, 64], stage=6, block='a')
    self.l6b = id_block([128, 128, 64], stage=6, block='b')
    self.l6c = id_block([128, 128, 64], stage=6, block='c')
    self.l6d = id_block([128, 128, 64], stage=6, block='d')

    self.l7a = convt_block([64, 64, OUTPUT_CHANNELS], stage=7, block='a')
    self.l7b = id_block([64, 64, OUTPUT_CHANNELS], stage=7, block='b')
    self.l7c = id_block([64, 64, OUTPUT_CHANNELS], stage=7, block='c')

  @tf.contrib.eager.defun
  def call(self, input_tensor, training):
    x = self.conv1(input_tensor)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)

    x1 = self.l2a(x, training=training)
    x = self.l2b(x1, training=training)
    x = self.l2c(x, training=training)

    x2 = self.l3a(x, training=training)
    x = self.l3b(x2, training=training)
    x = self.l3c(x, training=training)
    x = self.l3d(x, training=training)

    x3 = self.l4a(x, training=training)
    x = self.l4b(x3, training=training)
    x = self.l4c(x, training=training)
    x = self.l4d(x, training=training)
    x = self.l4e(x, training=training)
    x = self.l4f(x, training=training)

    x4 = tf.concat([x, x3], axis=-1)
    x4 = tf.contrib.layers.dropout(x4, keep_prob=0.5)
    x4 = self.l5a(x, training=training)
    x = tf.contrib.layers.dropout(x4, keep_prob=0.5)
    x = self.l5b(x4, training=training)
    x = self.l5c(x, training=training)
    x = self.l5d(x, training=training)

    x5 = tf.concat([x, x2], axis=-1)
    x5 = tf.contrib.layers.dropout(x5, keep_prob=0.5)
    x5 = self.l6a(x, training=training)
    x = tf.contrib.layers.dropout(x5, keep_prob=0.5)
    x = self.l6b(x5, training=training)
    x = self.l6c(x, training=training)
    x = self.l6d(x, training=training)

    x6 = tf.concat([x, x1], axis=-1)
    x6 = tf.contrib.layers.dropout(x6, keep_prob=0.5)
    x6 = self.l7a(x, training=training)
    x = tf.contrib.layers.dropout(x6, keep_prob=0.5)
    x = self.l7b(x6, training=training)
    x = self.l7c(x, training=training)

    x = tf.nn.tanh(x)
    return x
