# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: (./pics/center_pic.jpg "center lane picture?raw=true)  
[image2]: ./pics/aug_pic.jpg "flipped view"  
[image3]: ./pics/crop_pic.jpg "cropped image"  
 
![image1](./pics/aug_pic.jpg?raw=true)

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
Also tarining data is splitted in "tarining data" and "validation data" for both pair of picture and steering angle. 20% of data is used as validation data set, which are randomly seteclted by each trainig epoch. 

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


For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create simple network first, that can run simulation in autonomous mode. Then keep stepping up network to complecated one until the car can drive one lap by itself.

I have started with sigle neural network that only have one fully connected layer. This model has made in order to confirm all process works OK. Car did go off the road in first curve. 

Then I have tried modified LeNet. I thought this model might be appropriate because it gave pretty good result on picture classification. The modification had made only on input and dropout. Input has been changed from 32x32x1 to 160x320x3, and dropout(50%) is added in between each fully connected layer. Dropout is added in order to reduce risk of overfitting and create robust model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

LeNet workd OK for first curve, but then stacked on the curb. This model did not give enough steering angle. 

Then I tried different network. I have copied Nvidia end to end learning model, which designed to predict steering angle from image input. The network in the article did not have (or did not show) dropout in between any layer, so I have added in order to reduce overfitting. 


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture shown in model.py In[16] portion. This has following layers. There are 4 convolution layer, and 4 fully connected layer. The final layer output steering angle prediction given from input picture.

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

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 1 lap on track one using center lane driving. I just drive vehicle on center along road. This trainig data did not give good driving. Vehicle went off in first straing road by vehicle itself. 


I then changed driving method. give no steering input until car goes close to curb. if car become close to curb, then give steering angle to correct vehicle direction. This method should give good conbination of picture and given steering angle, means if car is on the left of the lane, rotate tire to left. I have tried to reduce "steering angle is on right when car is left" situation.

I have driven on clockwise and counter clockwise , once for each direction. I thought I need more lap to create enough amount of data, however 2 laps are (at least) enough to go around the cource so I have not added data. More data may help for more complecated cource. 

To augment the data sat, I also flipped images and angles thinking that this will help to normalize data. For example, below are images of original picture and flipped picture. When picture is flipped, steering angle is also flipped positive and negative.

Original data:
![](./pics/center_pic.jpg?raw=true)

Flipped picture:
![](./pics/aug_pic.jpg?raw=true)

Also right camera and left camera data are also used, with +0.2 and -0.2 steering angle.  PLease see In[8] portion of model.py  
right camera data is used as representative image of "go to right" signal. So steering angle (y_train of the image) is added +0.2 on original steering angle. In the same way, steering angle on left camera data has -0.2 on original angle.  

After the collection process, I had 10506 data points, means X_train shape is (10506, 160,320, 3)

I then preprocessed this data by trimming image from 160x320 to 65x320, so that I can remove unnecessary area, which are sky and hood.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

