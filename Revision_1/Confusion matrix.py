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
model_snapshot="train5000_val625_test_ALL_snapshot_rev1.h5"
test_dir = 'test_dir'

#load model
model = models.load_model(model_snapshot)

#generate the testing images
img_width,img_height = 224,224
batch_size = 25
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

#confusion Matrix

cm = confusion_matrix(test_data_generator.classes, y_pred)

#generate the diagram
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, figname,normalize=False,title="Confusionmatrix",cmap=plt.cm.Blues):
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment='center',color='white' if cm[i, j] > thresh else "black")
    
    plt.ylabel("True label")
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(figname)

#the names of the nine classes
category_names = sorted(os.listdir('train_dir'))
category_names=category_names[1:]

#plot the confusion matrix and save it
plot_confusion_matrix(cm, classes = category_names, title='Confusion Matrix', normalize=False, figname = 'Confusion_matrix_concrete.jpg')

