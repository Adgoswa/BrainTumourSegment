import numpy as np
from tqdm import tqdm
import nibabel as nib
import os
from Preprocess import *

# File path:
path = '/home/ag360/Code/DataSet/'

# read the mri image and return the numpy array of it
def read_image(file_path):
    img = nib.load(file_path)
    # the model to which this is being integrated reads the data in form (155, 240, 240), hence the transpose method
    img_data = img.get_fdata()
    return np.transpose(img_data, (2,1,0))

# iterate through the dataset, fetch the numpy array of all the MRI, normalize them, extract the patch and store it as .npy file
def load_data(path):

    my_dir = sorted(os.listdir(path))

    # specify the number of records in the dataset 
    limit = 369
    index = 0

    for p in tqdm(my_dir):
        gt = []
        x_image = np.zeros((4,155,240,240), dtype=np.float16)
        if ('.csv' not in p) and (index < limit):

            index += 1

            data_list = sorted(os.listdir(path+p))

            # FLAIR images:

            # FLAIR images:
            img = read_image(path + p + '/' + data_list[0])
            x_image[0,:,:,:,] = normalize_image(img)

            # Ground truth images:
            img = read_image(path + p + '/' + data_list[1])
            seg = pre_process(img)

            # T1 images:
            img = read_image(path + p + '/'+ data_list[2])
            x_image[1,:,:,:,] = normalize_image(img)

            # T1ce (T1Gd) images:
            img = read_image(path + p + '/' + data_list[3])
            x_image[2,:,:,:,] = normalize_image(img)

            # T2 images:
            img = read_image(path + p + '/' + data_list[4])
            x_image[3,:,:,:,] = normalize_image(img)

            x_image_slice = x_image[:,30:120, 60:188, 60:188]
            x_image_slice = np.transpose(x_image_slice, (1,2,3,0))

            #print(data.shape)

            gt = np.asarray(seg, dtype = np.float16)

            #print(gt.shape)

            # Saving the final data sets to the current directory:
            img_name = './SplitDataSet/Data_4/x_' + str(index) +'_img.npy'
            gt_name = './SplitDataSet/Label_4/y_' + str(index) +'_gt.npy'
            #print('x_image_slice', x_image_slice.shape)
            np.save(img_name, x_image_slice)
            np.save(gt_name, gt)


load_data(path)