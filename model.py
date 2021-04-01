import pandas as pd
import numpy as np
import cv2
import csv
from math import ceil
from scipy import ndimage
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.models import Model

#load csv data
def load_data():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)
    return lines
            
#randomly adjusting brightness of an image
def change_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_ratio = np.random.uniform(0.2,0.8)
    hsv_image[:,:,2] = hsv_image[:,:,2]*bright_ratio
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)         

# converting rgb to yuv channel as suggested in Nvidia model
def rgbtoyuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

#generate augmented image 
def augmentation(images,measurements):
    augmented_images , augmented_measurements = [],[]
    for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        #flipped image and its measurements
        flipped_image = cv2.flip(image,1)
        flipped_measurement = measurement * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)
        #change brightness and convert to yuv channel
        bright_image = rgbtoyuv(change_brightness(image))
        bright_flipped_image = rgbtoyuv(change_brightness(flipped_image))
        #append brightness changed images
        augmented_images.append(bright_image)
        augmented_measurements.append(measurement)
        augmented_images.append(bright_flipped_image)
        augmented_measurements.append(flipped_measurement)
    return augmented_images, augmented_measurements

#defining model network - Modified Nvidia model
def model_network_layers():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24,5,5, subsample = (2,2), activation = "relu"))
    model.add(Conv2D(36,5,5, subsample = (2,2), activation = "relu"))
    model.add(Conv2D(48,5,5, subsample = (2,2), activation = "relu"))
    model.add(Conv2D(64,3,3,activation = "relu"))
    model.add(Conv2D(64,3,3,activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def generator(samples,istraining, batch_size=32):
    num_samples = len(samples)
    correction = 0.4
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = ndimage.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                if istraining:  # only add left and right images for training data
                    left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                    img_left = cv2.imread(left_name)
                    right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                    img_right = cv2.imread(right_name)

                    images.append(img_left)
                    angles.append(center_angle + correction)
                    images.append(img_right)
                    angles.append(center_angle - correction)
            augmented_images, augmented_measurements = augmentation(images, angles)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)
            
samples = load_data()
print(len(samples))
# Set our batch size and epochs
batch_size=32
epochs = 5

# split data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# compile and train the model using the generator function
train_generator = generator(train_samples, istraining=True, batch_size=batch_size)
validation_generator = generator(validation_samples, istraining=False, batch_size=batch_size)

# define the model network
model = model_network_layers()
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
checkpoint = ModelCheckpoint('model-{epoch:02d}.h5',
                                 monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto')

model.compile(loss='mse', optimizer='adam')

#training model
model.fit_generator(train_generator,steps_per_epoch=ceil(len(train_samples)/batch_size),
            nb_epoch=epochs, 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            verbose=1, callbacks=[early_stopping, checkpoint])

#saving model
model.save('model.h5')
