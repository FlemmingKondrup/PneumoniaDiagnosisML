import numpy as np
from pathlib import Path
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

# STEP 1: Importing and preparing the data

#We get the paths to the training, testing and validation data folders
training_set_path = 'train'
testing_set_path = 'test'
val_set_path = 'val'

#Function will get x and y lists with respectively imaages and labels
def get_data(path):
    labels = ['NORMAL', 'PNEUMONIA']
    x = []
    y = []
    for label in labels:
        images_path = os.path.join(training_set_path,label)
        list_of_images = Path(images_path).rglob('*.jpeg')
        if label == 'NORMAL':
            for image in list_of_images:
                img = cv2.imread(os.path.join(image),cv2.IMREAD_GRAYSCALE)
                x.append(cv2.resize(img, (200,200)))
                y.append(0)
        elif label == 'PNEUMONIA':
            for image in list_of_images:
                img = cv2.imread(os.path.join(image), cv2.IMREAD_GRAYSCALE)
                x.append(cv2.resize(img, (200,200)))
                y.append(1)
    x = np.array(x)/255
    x = np.expand_dims(x, -1)
    y = np.array(y)/255
    return x,y

#We can now get the images and labels for each set (train, test and val)
train_x, train_y = get_data(training_set_path)
test_x, test_y = get_data(testing_set_path)
val_x,val_y = get_data(val_set_path)


# STEP 2: Build the CNN model & Fit it to the data
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', input_shape=(200, 200, 1)))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#may need a dropout() - regularization (against overfitting)

model.summary()
model.compile(optimizer = 'adam' , loss = tf.keras.losses.binary_crossentropy , metrics = ['accuracy'])

# STEP 3: Fit the data & Evaluate the model

history = model.fit(train_x, train_y, epochs=1, validation_data=(test_x, test_y)) #May need to optimize epochs

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print("Accuracy: " , test_acc)

