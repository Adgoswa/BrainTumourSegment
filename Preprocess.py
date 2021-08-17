import numpy as np
from tqdm import tqdm
import nibabel as nib
import os
from keras.utils.np_utils import to_categorical

def normalize_image(img):  
    # Normalize the image
    mean = img.mean()
    std = img.std()
    return (img - mean) / std

# extract the patch for the ground truth image, and change the labels to one hot encoding.
def pre_process(Y):
    Y = Y[30:120, 60:188, 60:188]
    Y = ground_truth_4_to_3(Y)
    Y = to_categorical(Y, num_classes=4)
    return Y

def ground_truth_4_to_3(label):
    # The groundTruth4to3 function relabels the ground truth class 4 pixels as class 3. Class 4 represents the
    # enhancing tumour region, which is then changed to class 3. This change is done to enable the one-hot-encoding
    # of the ground truth labels.
    label[np.where(label == 4)] = 3
    return label