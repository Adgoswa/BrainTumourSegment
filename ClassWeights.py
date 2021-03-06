import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import nibabel as nib
import os
from tensorflow.keras import backend as K
import pickle

# File path:
path = '/home/ag360/Code/DataSet/'
label_path = './SplitDataSet/Label/'

def read_image(file_path):
    img = nib.load(file_path)
    # the model to which this is being integrated reads the data in form (155, 240, 240), hence the transpose method
    img_data = img.get_fdata()
    return np.transpose(img_data, (2,1,0))

# read the entire records for the ground truth (GT) only and store them as .npy file, so that the number of classes for each GT can be calculated. 
def read_data(path):

    my_dir = sorted(os.listdir(path))

    # set the number of records in the dataset. 
    limit = 369
    index = 0

    for p in tqdm(my_dir):

        #data = []
        gt = []

        if ('.csv' not in p) and (index < limit):

            index += 1

            data_list = sorted(os.listdir(path+p))

            # Ground truth images:
            seg = read_image(path + p + '/' + data_list[1])

            gt = np.asarray(seg, dtype = np.uint8)

            gt_name = './SplitDataSet/Label/y_' + str(index) +'_gt.npy'

            np.save(gt_name, gt)

def load_data(path):

    my_dir = sorted(os.listdir(path))

    dataset = []

    for f in tqdm(my_dir):
        print(f)
        data = np.load(path + f)
        print(data.shape)
        dataset.append(data)

        
    dataset_conv = np.asarray(dataset)
    return dataset_conv

# calculate the number of classes for each of the tumour types in the GT images and store it as .pkl file, to be used during model training for loss calc.
def get_weights(dataset):

    print('dataset shape: ' + str(dataset.shape))

    flat_dataset = K.flatten(dataset)

    print('dataset shape: ' + str(flat_dataset.shape))

    label_count = np.bincount(flat_dataset)

    weight_0 = (1/label_count[0]) * (dataset.size*2)
    weight_1 = (1/label_count[1]) * (dataset.size*2)
    weight_2 = (1/label_count[2]) * (dataset.size*2)
    weight_3 = (1/label_count[4]) * (dataset.size*2)

    weight_dictionary = {0:weight_0, 1:weight_1, 2:weight_2 , 3:weight_3}

    print('Weight for class 0 (Background): ' + str(weight_0))
    print('Weight for class 1 (Necrotic Core): '+ str(weight_1))
    print('Weight for class 2 (Edema): '+ str(weight_2))
    print('Weight for class 3 (Enhancing Tumour): '+ str(weight_3))

    dict_file = open("class_weights.pkl", "wb")
    pickle.dump(weight_dictionary, dict_file)
    dict_file.close()


read_data(path)

data = load_data(label_path)

get_weights(data)
