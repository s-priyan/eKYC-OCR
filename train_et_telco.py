import matplotlib.pyplot as plt
import os, shutil,time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models,layers,optimizers

from termcolor import colored


model_weights_src = "/home/adldata/NIC_PROJECT/nic_project/weights/train5000_val625_test_ALL/train5000_val625_test_ALL.h5"
model_snapshot_src = "/home/adldata/NIC_PROJECT/nic_project/weights/train5000_val625_test_ALL/train5000_val625_test_ALL_snapshot.h5"
num_classes = 9
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

img_width,img_height = 224,224

conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_height, 3))  # 3 = RGB

print(conv_base.summary())
conv_base.trainable = False
print(conv_base.summary())

model = models.Sequential([conv_base])
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()

# Compile model
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['acc'])

#----------------------------------extract features----------------------------
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 50

dataset_path = "/home/adldata/NIC_PROJECT/NEW_DATA/ALL"
train_dir = dataset_path + '/train_dir'
validation_dir = dataset_path + '/val_dir'
test_dir = dataset_path + '/test_dir'

train_data_generator = datagen.flow_from_directory(train_dir,
                                        target_size=(img_width,img_height),
                                        batch_size = batch_size)

val_data_generator = datagen.flow_from_directory(validation_dir,
                                        target_size=(img_width,img_height),
                                        batch_size = batch_size)
epochs = 60
# Train model
history = model.fit(x = train_data_generator,# train_labels,
                    epochs=epochs,
                    validation_data = val_data_generator)

# serialize weights to HDF5
model.save_weights(model_weights_src)
print("\n\nSaved model weights to disk\n\n")

model.save(model_snapshot_src)
print("\n\nSaved snap shot of complete model\n\n")

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
