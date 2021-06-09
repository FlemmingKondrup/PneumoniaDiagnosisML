import numpy as np
from pathlib import Path
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 1: Importing and preparing the data

#We get the paths to the training, testing and validation data folders
training_set_path = 'train'
testing_set_path = 'test'
val_set_path = 'val'

#Function will get x and y lists with respectively imaages and labels
labels = ['NORMAL', 'PNEUMONIA']
def get_data(path):
    x = []
    y = []
    for label in labels:
        images_path = os.path.join(path,label)
        list_of_images = Path(images_path).rglob('*.jpeg')
        if label == 'NORMAL':
            for image in list_of_images:
                img = cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
                x.append(cv2.resize(img, (100,100)))
                y.append(0)
        elif label == 'PNEUMONIA':
            for image in list_of_images:
                img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
                x.append(cv2.resize(img, (100,100)))
                y.append(1)
    x = np.array(x)
    x = np.expand_dims(x, -1)
    y = np.array(y)
    return x,y

#We can now get the images and labels for each set (train, test and val)
train_x, train_y = get_data(training_set_path)
test_x, test_y = get_data(testing_set_path)
val_x,val_y = get_data(val_set_path)

# We normalize the data (0-255 grayscale -> 0-1)
train_x = np.array(train_x)/ 255
val_x = np.array(val_x) / 255
test_x = np.array(test_x) / 255

#We then perform data augmentation to avoid overfitting
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False
)

datagen.fit(train_x)

# STEP 2: Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', input_shape=(100, 100, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer = 'adam' , loss = tf.keras.losses.binary_crossentropy , metrics = ['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

# STEP 3: Fit the data & Evaluate the model

history = model.fit(datagen.flow(train_x, train_y,batch_size=32), verbose=2, epochs=12, validation_data=(val_x, val_y),
                    callbacks=[learning_rate_reduction])

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print("Accuracy: " , test_acc)
'''Accuracy:  0.909'''

