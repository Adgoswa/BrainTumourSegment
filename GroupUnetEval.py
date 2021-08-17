#Authors: DS291 and JH

import numpy as np
#from DiceScore_bin import *
from DiceScore import *
from sklearn.metrics import confusion_matrix, classification_report
#import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from PIL import Image
import pickle

# 
# The functions below are adapted from the Github Repository https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation, 
# accessed on 01/07/2021

################################################################### Loading Validation Data ###############################################################
#Method loads in the validation/ test data depending on the value passed in dataset param data.
def load_data(Data_path, Label_path, id_path, dataset):
    X_val = np.load(id_path + "X_" + dataset + ".npy")
    Y_val = np.load(id_path + "Y_" + dataset + ".npy")

    print(dataset + "dataset size is: ")
    print(X_val.shape)

    val_idx = X_val
    X_val = np.empty((len(val_idx),) + (90,128,128,4), dtype=np.float32) # this will be of the form (1, 4, 240, 240,155)
    Y_val = np.empty((len(val_idx),) + (90,128,128,4), dtype=np.float32) # this will be of the form (1, 240, 240,155)

    for i, ID in enumerate(val_idx):
        X_val[i,] = np.load(Data_path  + 'x_' + str(ID) + '_img.npy')
    X_val = X_val.reshape([-1,128,128,4])

    for i, ID in enumerate(val_idx):
        Y_val[i,] = np.load(Label_path + 'y_' + str(ID) + '_gt.npy')
    Y_val = Y_val.reshape([-1,128,128,4])

    return X_val, Y_val

################################################################### Output Images ###############################################################
#Puts one image over another
def overlay(slice1, slice2, index):
    #The ground truth slice should be the second slice
    plt.axis('off')
    plt.imshow(slice1, cmap=plt.cm.get_cmap('gray'))
    plt.savefig("brain.png", bbox_inches='tight')

    plt.axis('off')
    plt.imshow(slice2, cmap=plt.cm.get_cmap('gnuplot', 4))
    plt.savefig("seg.png", bbox_inches='tight')

    #After saving the images as .png, reopen as an image object to overlay 
    t1 = Image.open('brain.png')
    t2 = Image.open('seg.png')

    t1 = t1.convert("RGBA")
    t2 = t2.convert("RGBA")

    #Creates the blended image
    new_img = Image.blend(t1, t2, 0.5)
    new_img.save("./overlays/overlay" + str(index) + ".png","PNG")

Data_path = './SplitDataSet/Data_4/'
Label_path = './SplitDataSet/Label_4/'
id_path = './dataset_conv/'
X_val, Y_val = load_data(Data_path, Label_path, id_path, "val")
X_test, Y_test = load_data(Data_path, Label_path, id_path, "test")

# Load in model:
weights_file = open("class_weights.pkl", "rb")
weights = pickle.load(weights_file)
weights_file.close
weights_list = list(weights.values())

# Load in model:
unet_model = tf.keras.models.load_model('./models/group-UNet.h5',
                                            custom_objects={'lossFunc': weighted_loss(dice_loss_function, weights_list),
                                                            'dice_function': dice_function})

################################################################### Validation Predictions ###############################################################
# Get predictions from validation data using model
val_Y_pre = np.argmax(unet_model.predict(X_val), axis=-1)
Y_val = np.argmax(Y_val, axis=-1)

#Flatten the predictions and labels for the classification report.
oneD_val_pre = K.flatten(val_Y_pre)
oneD_Y = K.flatten(Y_val)

report = classification_report(oneD_Y, oneD_val_pre)

# The prediction array and Y_val array are reshaped:
val_Y_pre = val_Y_pre.reshape(-1, 128, 128, 1)
Y_val_reshape = Y_val.reshape(-1, 128, 128, 1)

# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores using validation data: ", dice_function_loop(Y_val_reshape, val_Y_pre))
print(report)

############################################################## Test set predictions ##############################################################
test_Y_pre = np.argmax(unet_model.predict(X_test), axis=-1) #X_test, Y_test
Y_test = np.argmax(Y_test, axis=-1)

#Flatten the predictions and labels for the classification report.
oneD_test_pre = K.flatten(test_Y_pre)
oneD_Y = K.flatten(Y_test)

report = classification_report(oneD_Y, oneD_test_pre)

# The prediction array and Y_test array are reshaped:
test_Y_pre = test_Y_pre.reshape(-1, 128, 128, 1)
Y_test_reshape = Y_test.reshape(-1, 128, 128, 1)

# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores using test data: ", dice_function_loop(Y_test_reshape, test_Y_pre))
print(report)


################################################################### Image saves ###############################################################
# The following loop takes slices 1490, 1500 and saves their corresponding X data, Y data, and the predicted segmentation.
X_val = X_val.astype('uint8')
val_Y_pre = val_Y_pre.astype('uint8')
Y_val_reshape = Y_val_reshape.astype('uint8')

for i in range(1490, 1500):
    overlay(X_val[i, :, :, 0], val_Y_pre[i, :, :, 0], (str(i) + '_Prediction'))
    overlay(X_val[i, :, :, 0], Y_val_reshape[i, :, :, 0], (str(i) + '_GroundTruth'))