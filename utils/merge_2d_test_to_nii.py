import glob2 as glob
import nibabel as nib
import os
import scipy.misc
import numpy as np

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
  for sli in range(num_slices):
    testi = scipy.misc.imread(os.path.join('isles18-dev', 'img2label_2d_ft_final_test',
                              os.path.basename(casei) + '_3ch_sli' + str(sli) + '.png'))
    outim[testi==255,sli] = 1
  
  fname = './test_2d/SMIR.generative_2d.' + smir_id + '.nii'
  outim_nii = nib.Nifti1Image(outim, affine=np.eye(4))
  outim_nii.set_data_dtype(ct.get_data_dtype())
  nib.save(outim_nii, fname)