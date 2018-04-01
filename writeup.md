# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/center_pic.jpg "center lane picture"
[image2]: ./pics/aug_pic.jpg "flipped view"
[image3]: ./pics/crop_pic.jpg "cropped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

Basically I have tried to make model and training data as simple as possible. less training, and less layer was my target. 

#### 1. Model architecture

My model is basically copy of Nvidia end to end learning model. here is reference
https://devblogs.nvidia.com/deep-learning-self-driving-cars/

I have added 50 % dropout between each fully connected layer in oder to make robust network.

As a preprocess, Picture is trimmed from 160x320 to 65x320, so that I can remove unnecessary area, which are sky and hood.
Then data is given to following layer

picture cropping  
original data:
[image1]

cropped data:
[image3]


Convolutional 5x5x24, then relu activation  
Convolutional 5x5x48, then relu activation  
Convolutional 3x3x64, then relu activation  
convolutional 3x3x64, then relu activation  
Flatten  
Fully connected layer to 100 nodes  
Dropout (50%)  
Fully connected layer to 50 nodes  
Dropout(50%)  
Fully connected layer to 10 nodes  
Dropout(50%)  
Fully connected layer to 1 nodes (that will be a steering angle output)  

#### 2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers in between each fulley connected layer in order to reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data containes 2 laps data. 1st lap is driven cownter clockwise, and 2nd lap is driven clocwise. both data was recorded with following method. -- give no steering input until car goes close to curb. if car become close to curb, then give steering angle to correct vehicle direction. This method should give good conbination of picture and given steering angle, means if car is on the left of the lane, rotate tire to left. I have tried to reduce "steering angle is on right when car is left" situation.

Also there are 3 points to give good training data to network. 
1) The training data are obtained with clockwise drive and counter clockwise drive.
2) Training data was preprocessed with augumentation method. (flip picture holizontally, and flip steering angle positive and negative)
3) right camera and left camera data are also used, with +0.2 and -0.2 steering angle.  PLease see In[8] portion of model.py

flipping picture
original data: [image1]  
flipped data : [image2]  
