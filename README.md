# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/nvidia_network.png "Nvidia Model"
[image2]: ./examples/sample_set.PNG "Sample set"
[image3]: ./examples/flipped.PNG "Flipped Image"
[image4]: ./examples/brightness.PNG "Brightness"
[image5]: ./examples/yuv.PNG "YUV"
[image6]: ./examples/cropped.PNG "Cropped Image"

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I've used what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I've trained,validated and tested Nvidia model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.


To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.



The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* model.py
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom and data can be collected from the simulator. For my project I've used dataset provided by udacity.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 track1
```

The fourth argument, `track1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls track1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py track1
```

Creates a video based on images found in the `track1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `track1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py track1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used slighty tweaked PilotNet CNN architecture published by NVIDIA. [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)

![alt text][image1]

My model consists of a Normalization layer,Cropping layer, 5 convolutional layers, a dropout layer and 4 fully connected layers.

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 

```
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
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 71). 

The model was trained and validated on 70% training data and 30% validation data.
```
train_samples, validation_samples = train_test_split(samples, test_size=0.3)
```
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by adding correction angel +/- 0.4 for left and right lanes respectively. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car without falling off the track.

My first step was to use a convolution neural network model similar to the PilotNet CNN provided by Nvidia as I thought this model might be appropriate because it is designed to make self driving cars to drive through various conditions without going off the track.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the the model in such a way that I have added Dropout layer after the Convolution layers.

Model checkpoint and early stop functionalities were used during training to chose best training model by monitoring the validation loss and stopping the training if the loss does not reduce in three consecutive epochs (model.py lines 128-130).The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I included left and right images by adding correction angle +/- 0.4 to the center steering angle. I have also cropped all the images 70px from top and 25px from bottom by adding Cropping2D layer to crop sky, trees,hills, hood of the car. I have also augmented and preprocessed the images which will be discussed below in order to enhance the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-76) consisted of a convolution neural network with the following layers and layer sizes:

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
_________________________________________________________________
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To train the model, I have used Udacity Provided dataset. Below is the Samples from the dataset:

![alt text][image2]

I have carried out the whole data augmentation and preprocessing steps inside the generator function (model.py line 79). In order to work with manageable batch of dataset without loading the whole dataset into memory, generator functionality is used which improves efficiency when working with large volumes of data.

In case of Image Augmentation,

1. I initially Flipped the images and changed the sign of its steering angle which is as same as driving the car in opposite direction which also increases the amount of data. Below is an example of flipped images:

![alt text][image3]

2. After this, I altered the brightness of the image by randomly scaling the V channel by converting image to HSV channel. Below is an example:

![alt text][image4]

3. After this step I converted the images to YUV channel as suggested in Nvidia's paper. Below is the result:

![alt text][image5]

I finally randomly shuffled and yielded the data set from generator (model.py line 109).

Further I have also Normalized the images by adding normalization layer to the network. And also cropped 70px from top and 25px from bottom by adding Cropping2D layer to the network. Below is an example of how keras cropping works taken from Udacity Lab:

![alt text][image6]

I used 70% of my dataset for training and 30% for validating.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by running my model again and again. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Result
Final Result is a video file 'track1.mp4' generated using video.py script which is used to make a video of the vehicle when it is driving autonomously.
