# brain-synthesis-lesion-segmentation
Code for the paper "Medical Image Synthesis for Data Augmentation and Anonymization using Generative Adversarial Networks" and NVIDIA DLI workshop on GAN for Medical Imaging

https://arxiv.org/abs/1807.10225

The original paper used the [BRATS'15 Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html) dataset and [ADNI](http://adni.loni.usc.edu/) dataset, and [PyTorch](https://pytorch.org/).
This repository has been modified for NVIDIA DLI workshop on GAN for Medical Imaging, to use [ISLES'18 Challenge](http://www.isles-challenge.org/) dataset and [TensorFlow](https://www.tensorflow.org/).

The method is based on [pix2pix](https://phillipi.github.io/pix2pix/) and the code is based on its [TensorFlow implementation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/pix2pix).
