#Authors: DS291

# The functions below are adapted from the Github Repository https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation, 
# accessed on 01/07/2021

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization, concatenate, Input, Dropout,Conv2DTranspose, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from DiceScore_bin import *
from DiceScore import *
from tqdm import tqdm
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from DataGenerator import DataGenerator

########################################################################################################################################################
#
#                                                           U-NET MODEL ARCHITECTURE:
#
#
#                                       Input                                                           Output
#                                           block1                                              decode_block4
#                                               block2                                      decode_block3
#                                                   dropout1                            dropout2
#                                                       block3                      decode_block2
#                                                           block4             decode_block1
#                                                                   block_5
#
########################################################################################################################################################

#   Input Layer
input_ = Input(shape=(128, 128,4), name='input')

##########################################################      ENCODING PATH       ####################################################################

# Block Architecture
#   Two convolutional layers (Conv2D)
#   Normalisation function (BatchNormalisation)
#   Pooling layer (MaxPooling2D)

tf.random.set_seed(1234)

#   EXAMPLE ENCODER
block1_conv1 = Conv2D(16, 3, padding='same', activation='relu', name='block1_conv1')(input_)
block1_conv2 = Conv2D(16, 3, padding='same', activation='relu', name='block1_conv2')(block1_conv1)
block1_norm = tfa.layers.GroupNormalization(groups=8, axis=3)(block1_conv2)
block1_pool = MaxPooling2D(name='block1_pool')(block1_norm)

block2_conv1 = Conv2D(32, 3, padding='same', activation='relu', name='block2_conv1')(block1_pool)
block2_conv2 = Conv2D(32, 3, padding='same', activation='relu', name='block2_conv2')(block2_conv1)
block2_norm = BatchNormalization(name='block2_batch_norm')(block2_conv2)
block2_pool = MaxPooling2D(name='block2_pool')(block2_norm)

block3_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='block3_conv1')(block2_pool)
block3_conv2 = Conv2D(64, 3, padding='same', activation='relu', name='block3_conv2')(block3_conv1)
block3_norm = BatchNormalization(name='block3_batch_norm')(block3_conv2)
block3_pool = MaxPooling2D(name='block3_pool')(block3_norm)

block4_conv1 = Conv2D(128, 3, padding='same', activation='relu', name='block4_conv1')(block3_pool)
block4_conv2 = Conv2D(128, 3, padding='same', activation='relu', name='block4_conv2')(block4_conv1)
block4_norm = BatchNormalization(name='block4_batch_norm')(block4_conv2)
block4_pool = MaxPooling2D(name='block4_pool')(block4_norm)

#BOTTOM OF U
block5_conv1 = Conv2D(256, 3, padding='same', activation='relu', name='block5_conv1')(block4_pool)

##########################################################      DECODING PATH       ####################################################################

# Decoder Architecture
#   Transposed Convolution layer (Conv2DTranspose)
#   Concatenation layer (concatenate)
#   Convolution layer (Conv2D)


#   EXAMPLE DECODER

up_pool1 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', activation='relu', name='up_pool1')(block5_conv1)
merged_block1 = concatenate([block4_norm, up_pool1], name='merged_block1')

decod_block1_conv1 = Conv2D(128, 3, padding='same', activation='relu', name='decod_block1_conv1')(merged_block1)
up_pool2 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', activation='relu', name='up_pool2')(decod_block1_conv1)
merged_block2 = concatenate([block3_norm, up_pool2], name='merged_block2')

decod_block2_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='decod_block2_conv1')(merged_block2)
up_pool3 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', activation='relu', name='up_pool3')(decod_block2_conv1)
merged_block3 = concatenate([block2_norm, up_pool3], name='merged_block3')

decod_block3_conv1 = Conv2D(32, 3, padding='same', activation='relu', name='decod_block3_conv1')(merged_block3)
up_pool4 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same', activation='relu', name='up_pool4')(decod_block3_conv1)
merged_block4 = concatenate([block1_norm, up_pool4], name='merged_block4')
decod_block4_conv1 = Conv2D(16, 3, padding='same', activation='relu', name='decod_block4_conv1')(merged_block4)


##########################################################      OUTPUT       ####################################################################

# Output Architecture
#   pre-output convolutional layer
#   output convolutional layer
#   Model declaration (unet_model = Model(inputs, outputs))

# EXAMPLE

pre_output = Conv2D(16, 1, padding='same', activation='relu', name='pre_output')(decod_block4_conv1)

output = Conv2D(4, 1, padding='same', activation='softmax', name='output')(pre_output)

modelUNet = Model(inputs=input_, outputs=output)
print(modelUNet.summary())

##########################################################      EXECUTION        ###############################################################

# Generator function call for loading bulk data in batches
image_path = './SplitDataSet/Data_4/'
label_path = './SplitDataSet/Label_4/'
total_records = 369

#Lists with number of records
Xrecords = list(range(1,total_records))
Yrecords = list(range(1,total_records))

#Creating numpy arrays that contain the IDs of training/validation/test data
#X_train, X_test, Y_train, Y_test = train_test_split(Xrecords, Yrecords, test_size=0.10, random_state=1234)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1234)
#X_train, X_val, Y_train, Y_val = train_test_split(Xrecords, Yrecords, test_size=0.10, random_state=1234)
#X_train, X_test, Y_train, Y_test = train_test_split(Xrecords, Yrecords, test_size=0.10, random_state=1234)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=1234)

X_train, X_val, Y_train, Y_val = train_test_split(Xrecords, Yrecords, test_size=0.10, random_state=1234)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.20, random_state=1234)

train_idx = X_train
val_idx = X_val

#Saving test data for Evaluation
np.save("./dataset_conv/X_train", X_train)
np.save("./dataset_conv/Y_train", Y_train)
np.save("./dataset_conv/X_val", X_val)
np.save("./dataset_conv/Y_val", Y_val)
np.save("./dataset_conv/X_test", X_test)
np.save("./dataset_conv/Y_test", Y_test)

training_generator = DataGenerator(train_idx, image_path, label_path)
validation_generator = DataGenerator(val_idx, image_path, label_path)

weights_file = open("class_weights.pkl", "rb")
weights = pickle.load(weights_file)
weights_file.close
weights_list = list(weights.values())

print(weights_list)

#Process
# Compile model with model.compile(optimiser, loss, metrics)
# Apply early stopping with callbacks.EarlyStopping(patience, monitor)
# Fit the model with Model.fit(x,y,validation_data, batch_size, epochs, shuffle, callbacks)
# Save the model with model.Save(path, overwrite)

#EXAMPLE

# The model is compiled with the dice_loss_function and the dice_function metric:
print("About to compile...")
#modelUNet = Sequential()

modelUNet.compile(optimizer=Adam(learning_rate=(5*1e-5)), loss=weighted_loss(dice_loss_function, weights_list), metrics=[dice_function])

# EarlyStopping is applied incase the model stops improving with each epoch:
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')]


history = modelUNet.fit(
	x=training_generator,
	validation_data=validation_generator,
	epochs=100, callbacks=callbacks)

modelUNet.save('./models/group-UNet.h5', overwrite=True)
print("ModelSaved Successfully")

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['dice_function'])
plt.plot(history.history['val_dice_function'])
plt.title('Model Accuracy')
plt.ylabel('Dice_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('GroupUnet_Accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Dice Loss')
plt.ylabel('Dice_loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('GroupUnet_Loss.png')
