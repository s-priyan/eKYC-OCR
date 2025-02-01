#test the dataset
import tensorflow as tf 
import numpy as np
import base64
import cv2
import matplotlib.pyplot as plt
import os, shutil, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from termcolor import colored
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import models,layers,optimizers

#Check whether the GPUs are initialized (output:"1 Physical GPUs, 1 Logical GPUs")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#path to model snapshot and test set
model_snapshot="train5000_val625_test_ALL_snapshot_rev1_new.h5"
test_dir = 'test_new_dir'

#load model
model = models.load_model(model_snapshot)

#generate the testing images
img_width,img_height = 224,224
batch_size = 16
num_classes = 9
datagen = ImageDataGenerator(rescale=1./255)
test_data_generator = datagen.flow_from_directory(test_dir,
                                        target_size=(img_width,img_height),
                                        batch_size = batch_size,class_mode = "categorical",shuffle=False)
#Prediction
Y_pred = model.predict_generator(test_data_generator)
y_pred = np.argmax(Y_pred, axis=1)

#accuracy
accuracy = accuracy_score(test_data_generator.classes, y_pred)
print("Accuracy in test set: %0.1f%% " % (accuracy * 100))
