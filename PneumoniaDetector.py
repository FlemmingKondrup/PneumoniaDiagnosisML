import numpy as np
from pathlib import Path
import cv2
import os

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
    return np.array(x),np.array(y)

#We can now get the images and labels for each set (train, test and val)
train_x, train_y = get_data(training_set_path)
test_x, test_y = get_data(testing_set_path)
val_x,val_y = get_data(val_set_path)

