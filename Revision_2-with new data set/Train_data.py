#train the dataset
#import necessary libraries
import matplotlib.pyplot as plt
import os, shutil,time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models,layers,optimizers
from termcolor import colored

#Path to the files where the weights and the model should be saved
model_weights_src = "train5000_val625_test_ALL_rev1_new.h5"
model_snapshot_src = "train5000_val625_test_ALL_snapshot_rev1_new.h5"

#Path to the datasets
train_dir ='train_new_dir'
validation_dir = 'val_new_dir'
test_dir ='test_new_dir'

#variables
num_classes = 9
img_width,img_height = 224,224
batch_size = 16
epochs = 60

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
        
#the base will be the VGG16 model used for imagenet with max pooling
conv_base = VGG16(weights='imagenet', 
                  include_top=False, pooling="max",
                  input_shape=(img_width, img_height, 3))  # 3 = RGB

print(conv_base.summary())

#add a dense layer to narrow for the nine classes
model = models.Sequential([conv_base])
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()

#compile the model
model.compile(optimizer=optimizers.Adam(lr=5e-5, clipnorm = 1.),
              loss='categorical_crossentropy',
              metrics=['acc'])

#scaling
datagen = ImageDataGenerator(rescale=1./255)

#generating the data
train_data_generator = datagen.flow_from_directory(train_dir,
                                        target_size=(img_width,img_height),
                                        batch_size = batch_size)

val_data_generator = datagen.flow_from_directory(validation_dir,
                                        target_size=(img_width,img_height),
                                        batch_size = batch_size)

#learning the model
history = model.fit(x = train_data_generator,
                    epochs=epochs,
                    validation_data = val_data_generator)

#save weights and model snapshot
model.save_weights(model_weights_src)
print("\n\nSaved model weights to disk\n\n")

model.save(model_snapshot_src)
print("\n\nSaved snap shot of complete model\n\n")

#plots for training and validation accuracy & loss Vs the num of epochs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

