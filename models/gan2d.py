import tensorflow as tf


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc1 = tf.keras.layers.Dense(8*8*64, use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.conv1 = tf.keras.layers.Conv2DTranspose(64 * 8, (4, 4), strides=(1, 1), padding='same', use_bias=False)
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(64 * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm3 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2DTranspose(64 * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm4 = tf.keras.layers.BatchNormalization()

    self.conv4 = tf.keras.layers.Conv2DTranspose(64 * 1, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm5 = tf.keras.layers.BatchNormalization()

    self.conv5 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='same', use_bias=False)

  def call(self, x, training=True):
    x = self.fc1(x)
    x = self.batchnorm1(x, training=training)
    x = tf.nn.relu(x)

    x = tf.reshape(x, shape=(-1, 8, 8, 64))

    x = self.conv1(x)
    x = self.batchnorm2(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batchnorm3(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv3(x)
    x = self.batchnorm4(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv4(x)
    x = self.batchnorm5(x, training=training)
    x = tf.nn.relu(x)

    x = tf.nn.tanh(self.conv5(x))  
    return x


class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
    self.conv2 = tf.keras.layers.Conv2D(64 * 2, (4, 4), strides=(2, 2), padding='same')
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    self.conv3 = tf.keras.layers.Conv2D(64 * 4, (4, 4), strides=(2, 2), padding='same')
    self.batchnorm3 = tf.keras.layers.BatchNormalization()
    self.conv4 = tf.keras.layers.Conv2D(64 * 8, (4, 4), strides=(2, 2), padding='same')
    self.batchnorm4 = tf.keras.layers.BatchNormalization()
    self.conv5 = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid')

    self.dropout = tf.keras.layers.Dropout(0.3)
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(1)

  def call(self, x, training=True):
    x = tf.nn.leaky_relu(self.conv1(x))

    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    x = tf.nn.leaky_relu(x)

    x = self.conv3(x)
    x = self.batchnorm3(x, training=training)
    x = tf.nn.leaky_relu(x)

    x = self.conv4(x)
    x = self.batchnorm4(x, training=training)
    x = tf.nn.leaky_relu(x)

    x = self.conv5(x)
    x = self.fc1(x)
    return x


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)
