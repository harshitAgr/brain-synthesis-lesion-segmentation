from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time

from absl import flags
import tensorflow as tf
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from data import gan2d_loader
from models import gan2d

tfe = tf.contrib.eager


BATCH_SIZE = 256
EPOCHS = 150
noise_dim = 100
num_examples_to_generate = 16


def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))
  
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
        
  # tight_layout minimizes the overlap between 2 sub-plots
  plt.tight_layout()
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


def train(dataset, epochs, noise_dim, generator, discriminator,
          generator_optimizer, discriminator_optimizer, random_vector_for_generation,
          checkpoint, checkpoint_prefix):  
  for epoch in range(epochs):
    start = time.time()
    
    for images in dataset:
      # generating noise from a uniform distribution
      noise = tf.random_normal([BATCH_SIZE, noise_dim])
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
      
        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)
        
        gen_loss = gan2d.generator_loss(generated_output)
        disc_loss = gan2d.discriminator_loss(real_output, generated_output)
        
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
      
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

      
    if epoch % 1 == 0:
      #display.clear_output(wait=True)
      generate_and_save_images(generator,
                               epoch + 1,
                               random_vector_for_generation)

    # saving (checkpoint) the model every 15 epochs
    if epoch % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))
  # generating after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           random_vector_for_generation)


def main(_):
  tf.enable_eager_execution()
  
  dataset = gan2d_loader.data_loader(FLAGS.data_dir)
  dataset = dataset.batch(BATCH_SIZE).prefetch(BATCH_SIZE * 4)

  generator = gan2d.Generator()
  discriminator = gan2d.Discriminator()

  generator.call = tf.contrib.eager.defun(generator.call)
  discriminator.call = tf.contrib.eager.defun(discriminator.call)

  discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
  generator_optimizer = tf.train.AdamOptimizer(1e-4)

  checkpoint_dir  = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])

  train(dataset, EPOCHS, noise_dim, generator, discriminator,
        generator_optimizer, discriminator_optimizer,
        random_vector_for_generation, checkpoint, checkpoint_prefix)

if __name__ == "__main__":
  flags.DEFINE_string(
    "data_dir", default='../datasets/adni/png', help="data set directory")
  FLAGS = flags.FLAGS
  tf.app.run(main)
