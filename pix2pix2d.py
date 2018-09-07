import tensorflow as tf

import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import PIL, matplotlib, random
import scipy.misc

from absl import flags
import glob2 as glob
from data import pix2pix2d_loader

from models import pix2pix2d

BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 200



PATCH_SIZE = [70, 70]
NUM_PATCHES = 1000

def sample_patches(input_image, generated_images, labels, sample_from):
  label_patches = []
  gen_patches = []

  if sample_from == 'input_image':
    nz_idxs = tf.where(tf.greater(input_image, -1))
  else:
    nz_idxs = tf.where(tf.greater(labels, 0))
  nz_idxs = nz_idxs.numpy().tolist()

  if sample_from == 'label' and not nz_idxs:
    nz_idxs = tf.where(tf.greater(input_image, -1))
    nz_idxs = nz_idxs[:100]
    nz_idxs = nz_idxs.numpy().tolist()

  if not nz_idxs:
    nz_idxs = tf.where(tf.equal(input_image, -1))
    nz_idxs = nz_idxs[:100]
    nz_idxs = nz_idxs.numpy().tolist()

  random.shuffle(nz_idxs)
  for count, nz_idxi in enumerate(nz_idxs):

    for aidx in range(1, 3):
      if nz_idxi[aidx] - np.ceil(PATCH_SIZE[aidx-1]/2) < 0:
        nz_idxi[aidx] += (np.ceil(PATCH_SIZE[aidx-1]/2) - nz_idxi[aidx])
      if nz_idxi[aidx] + np.ceil(PATCH_SIZE[aidx-1]/2) >= labels.shape[aidx]:
        nz_idxi[aidx] -= (nz_idxi[aidx] - np.ceil(PATCH_SIZE[aidx-1]/2))

      nz_idxi[aidx] -= np.ceil(PATCH_SIZE[aidx-1]/2)
      nz_idxi[aidx] = int(nz_idxi[aidx])

    nz_idxi[3] = 0
    label_patches.append(tf.slice(labels, nz_idxi, [1, PATCH_SIZE[0], PATCH_SIZE[1], 3]))
    gen_patches.append(tf.slice(generated_images, nz_idxi, [1, PATCH_SIZE[0], PATCH_SIZE[1], 3]))

    if count >= NUM_PATCHES:
      break

  label_patches = label_patches[:NUM_PATCHES]
  gen_patches = gen_patches[:NUM_PATCHES]
  label_patches = tf.concat(label_patches, axis=0)
  gen_patches = tf.concat(gen_patches, axis=0)

  return label_patches, gen_patches


def generate_images(model, test_input, tar, epoch, train_or_test):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)

  if train_or_test == 'test':
    prediction_ = prediction.numpy()[0,:,:,1]
    prediction_[prediction_ > 0.5] = 1
    prediction_[prediction_ <= 0.5] = 0
    scipy.misc.imsave(os.path.join(FLAGS.output_file_dir, epoch + '.png'), prediction_)
    return 0
    
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')

  if not os.path.exists(FLAGS.output_file_dir):
    os.makedirs(FLAGS.output_file_dir)
  if isinstance(epoch, int):
    plt.savefig(os.path.join(FLAGS.output_file_dir,
                'image_at_epoch_{:04d}.png'.format(epoch)))
  else:
    plt.savefig(os.path.join(FLAGS.output_file_dir,
                'image_at_epoch_' + epoch + '.png'))
  plt.show()
  plt.close()


def train(dataset, test_dataset, epochs, generator, discriminator,
          generator_optimizer, discriminator_optimizer,
          checkpoint, checkpoint_prefix):

  for epoch in range(epochs):
    start = time.time()

    for input_image, target in dataset:

      if FLAGS.swap_noise_imB_channel_13 and np.random.random() > 0.9:
        target= tf.reverse(target, [-1])
        
        noise_axis = [0, 2]
        target_0 = tf.slice(target, [0,0,0,0], [BATCH_SIZE, 256, 256, 1])
        target_1 = tf.slice(target, [0,0,0,1], [BATCH_SIZE, 256, 256, 1])
        target_2 = tf.slice(target, [0,0,0,2], [BATCH_SIZE, 256, 256, 1])
        target_l = [target_0, target_1, target_2]
        nxi = np.random.randint(2)
        target_l[noise_axis[nxi]] = tf.multiply(target_l[noise_axis[nxi]], tf.random_normal([1, 256, 256, 1],
                                                stddev=0.1))
        target_l[noise_axis[nxi-1]] = target_1
        target = tf.concat(target_l, axis=3)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator(input_image, target, training=True)
        disc_generated_output = discriminator(input_image, gen_output, training=True)

        if FLAGS.disc_dim_method == 'patch-based':
            label_patches_nzlabel, gen_patches_nzlabel =\
              sample_patches(input_image, gen_output, target, 'label')

            label_patches = label_patches_nzlabel 
            gen_patches = gen_patches_nzlabel 

            gen_loss = pix2pix2d.generator_loss(disc_generated_output, gen_patches, label_patches)
        else:
            gen_loss = pix2pix2d.generator_loss(disc_generated_output, gen_output, target)

        disc_loss = pix2pix2d.discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_loss,
                                              generator.variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.variables))

    if epoch % 1 == 0:
        #clear_output(wait=True)
        for inp, tar in test_dataset.take(1):
          generate_images(generator, inp, tar, epoch, FLAGS.train_or_test)

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


def main(_):
  listA = sorted(glob.glob(os.path.join(FLAGS.data_dir_A, '*.png')))
  listB = sorted(glob.glob(os.path.join(FLAGS.data_dir_B, '*.png')))

  assert len(listA) == len(listB)

  _listAB = list(zip(listA, listB))
  random.shuffle(_listAB)

  if FLAGS.test_data_dir_A == 'None':
    _trainC = _listAB[:int(np.round(len(_listAB)*0.9))]
    _testC = _listAB[int(np.round(len(_listAB)*0.9)):]
  else:
    t_listA = sorted(glob.glob(os.path.join(FLAGS.test_data_dir_A, '*.png')))
    t_listB = sorted(glob.glob(os.path.join(FLAGS.test_data_dir_B, '*.png')))
    _t_list_AB = list(zip(t_listA, t_listB))
    _trainC = _listAB
    _testC = _t_list_AB

  trainA, trainB = zip(*_trainC)
  testA, testB = zip(*_testC)
  trainA, trainB, testA, testB =\
    list(map(lambda x: list(x), [trainA, trainB, testA, testB]))

  trainA = tf.data.Dataset.from_tensor_slices(trainA)
  trainB = tf.data.Dataset.from_tensor_slices(trainB)
  trainAB = tf.data.Dataset.zip((trainA, trainB))
  train_dataset = trainAB.shuffle(BUFFER_SIZE)
  train_dataset = train_dataset.map(
                  lambda x, y: pix2pix2d_loader.load_image(x, y, True),
                  num_parallel_calls=16)
  train_dataset = train_dataset.batch(1).prefetch(4)

  testA = tf.data.Dataset.from_tensor_slices(testA)
  testB = tf.data.Dataset.from_tensor_slices(testB)
  testAB = tf.data.Dataset.zip((testA, testB))
  test_dataset = testAB.map(
                  lambda x, y: pix2pix2d_loader.load_image(x, y, False),
                  num_parallel_calls=16)
  test_dataset = test_dataset.batch(1).prefetch(4)

  if FLAGS.generator_type == 'unet':
    generator = pix2pix2d.Generator()
  else:
    generator = pix2pix2d.GeneratorRes()
  discriminator = pix2pix2d.Discriminator()

  generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
  discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)


  checkpoint_dir = os.path.join(FLAGS.output_file_dir, 'training_checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  if FLAGS.restore_checkpoints:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  if FLAGS.train_or_test == 'train':
    train(train_dataset, test_dataset, EPOCHS, generator, discriminator,
          generator_optimizer, discriminator_optimizer,
          checkpoint, checkpoint_prefix)


  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  for idx, (inp, tar) in enumerate(test_dataset):
    if FLAGS.train_or_test == 'train':
      fprefix = 'FINAL_' + str(idx)
    else:
      fprefix = t_listA[idx].split('/')[-1][:-4]
    generate_images(generator, inp, tar, fprefix, FLAGS.train_or_test)


if __name__ == "__main__":
  tf.enable_eager_execution()

  flags.DEFINE_string(
    "data_dir_A", default='../datasets/ISLES2018/training_png/img',
    help="data set A directory")
  flags.DEFINE_string(
    "data_dir_B", default='../datasets/ISLES2018/training_png/seg',
    help="data set B directory")
  flags.DEFINE_string(
     "test_data_dir_A", default='None', help='test dataset A directory')
  flags.DEFINE_string(
     "test_data_dir_B", default='None', help='test dataset B directory')
  flags.DEFINE_string(
     "output_file_dir", default='.', help='directory to save output images.')
  flags.DEFINE_string(
     "disc_dim_method", default='whole-image', help='whole-image (not sample) or patch-based (sample).')
  flags.DEFINE_string(
     "generator_type", default='unet', help='generator type unet/resnet')
  flags.DEFINE_string(
     "train_or_test", default='train', help='train or test')
  flags.DEFINE_boolean(
     "restore_checkpoints", default=False, help='whether to restore a checkpoint')
  flags.DEFINE_boolean(
     "swap_noise_imB_channel_13", default=False, help='whether to swap and add noise to the 1,3 channels of imageB')
  FLAGS = flags.FLAGS
  tf.app.run(main)
