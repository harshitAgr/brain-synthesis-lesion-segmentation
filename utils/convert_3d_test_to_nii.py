import glob2 as glob
import nibabel as nib
import os, pickle
import scipy.misc
import numpy as np

Z_DIM = 24

cases = glob.glob('../datasets/ISLES2018/TESTING/case_*')
for casei in cases:
  mods = glob.glob(os.path.join(casei, '*/'))
  
  for modi in mods:
    modi = os.path.normpath(modi)
    if '_MTT' in modi:
      smir_id = modi.split('.')[-1]

    if '.CT.' in modi:
      ct = nib.load(os.path.join(modi, os.path.basename(modi) + '.nii'))
    
  ctim = ct.get_data()
  num_slices = ctim.shape[2]
  outim = np.zeros((256, 256, num_slices), dtype=np.uint8)

  z_start = int(Z_DIM/2 - np.ceil(ctim.shape[2]/2))
  testim_fname = os.path.join('isles18-dev', 'img2label_3d_ft_final',
                              os.path.basename(casei) + '.pkl')
  with open(testim_fname, 'rb') as f:
    _testim = pickle.load(f)
  _testim = np.swapaxes(_testim, 0, 2)
  testim = _testim[:,:,z_start:z_start+ctim.shape[2]]
  outim[testim > 0] = 1

  fname = './test_3d/SMIR.generative_3d.' + smir_id + '.nii'
  outim_nii = nib.Nifti1Image(outim, affine=np.eye(4))
  outim_nii.set_data_dtype(ct.get_data_dtype())
  nib.save(outim_nii, fname)