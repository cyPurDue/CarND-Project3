# **Behavioral Cloning** 

## Writeup Report


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

image before cropping: <br />
[image1]: sample_images/before_cropping.jpg "Before cropping" <br />
image after cropping: <br />
[image2]: sample_images/after_cropping.jpg "After cropping" <br />

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 showing autonomous driving result (covered 1 full lap)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Data logging and pre-processing
Using the simulation software, data from 3 cameras are logged (mid, left, right). All data are saved in local folder (not uploaded here). For training set, only steering data is used, associated with corresponding captured pictures. To pre-process all captured pictures, the code reads the .csv file, parsing paths of pictures, and loads to local memory.

#### 2. Creating more training data
As learned in the class, using left and right cameras with reasonable adjustment can provide more usable data based on limited training data. I have followed the example and used steering factor of 0.2 to generate and append 2x more data based on left and right cameras. Note that this may lead to overload the memory, and in case of that, try to pick up a random camera out of three at one time. 

#### 3. Data augmentation
Data is augmented by using the flipping example in class. To show the result of cropping, I have randomly picked up an image and illustrated the results before and after cropping. <br />

![alt text][image1] <br />
![alt text][image2] <br />



#### 4. Training and validation
Here I have used 80% data for training, and 20% for validataion. 3 epoches are used every time, and I can see the training loss is decreasing all the time.

#### 5. Training model architecture
I used the end-to-end training model architecture published by NVidia [1], and here is the summary: <br />
Layer1: Lambda -> (160, 320, 3) <br />
Layer2: Cropping2D (modified to (50, 20), (0, 0)) -> (90, 320, 3) <br />
Layer3: Convolution2D, 24 filters, kernel (5,5), strides (2,2), relu activation <br />
Layer4: Convolution2D, 36 filters, kernel (5,5), strides (2,2), relu activation <br />
Layer5: Convolution2D, 48 filters, kernel (5,5), strides (2,2), relu activation <br />
Layer6: Convolution2D, 64 filters, kernel (3,3), strides (1,1), relu activation <br />
Layer7: Convolution2D, 64 filters, kernel (3,3), strides (1,1), relu activation <br />
Layer8: Dropout(0.35) <br />
Layer9: Flatten <br />
Layer10: Dense, output 100 <br />
Layer11: Dense, output 50 <br />
Layer12: Dense, output 10 <br />
Layer13: Dense, output 1 <br />

The model uses Adam optimizer, and the learning rate was not tuned. In order to prevent any overfitting, I have added one layer of dropout. Having tried a few parameters, I have used keep rate of 0.35 to train the model. The training and validation loss kept decreasing, and the car is driving smoothly.

#### 6. Obtaining better training data
After having model programmed and some data collected, I have tried to train the first set of data. The original data consists of 2x full good run of the track. It could run a few seconds on the road, but once it was biased, it never recovered and came back. After learning more exercise, especially [2] that the class recommended, I found out the reason is lack of recovery data collected. 

Therefore, I have added at least 1x recovery data, with seperately collected, to let the model know how to turn back to center quickly when it is offroad. Also as recommended, I have collected 1x run of reversed-direction driving. Using this set of data basically achieved much better results. The car can at least finish a full run of the track, but around 20% of time, it is either too biased (following side lines), or turn less on sharp turns. 

In order to fix this problem, I have added 1x very good run especially at those biased-driving locations. Then I repeated around those corner cases, and recorded several pieces of data. Good that this did not lead to overfit other places, and fixed the issue before. 

#### 7. Other thinking
This entire project provides a very comprehensive idea of how to use user data train the driving model, and the model recommended is not complicated to learn and working very well. Same as other posts that I have seen online, the hard part is to get useful data, and here are some thinking based on my experience:
i) Use mouse for steering. My feeling is using mouse provides more detailed resolution than keyboard.
ii) Since you are using mouse, when finished recording, click the record button asap, or stop the car ahead. Because the car is driving in tangent line (steering = 0) during the time from control to stop recording.
iii) Cropping is very very useful. 
iv) Try to finish a full run first, and then add more repeated good data to those corner cases.
v) For those corner cases, drive back and forth, not only 1 direction.
vi) One feedback for the simulation software: it is relatively easy to get stucked on the curb. Many times when I wanted to make U turn to obtain revserse-direction data, if the car speed is slow and car is crossing the curb, it stcuk there and I have to restart the software. But overall I really appreciate this awesome work provided for us.

#### Reference:
[1] Bojarski, M. et al., "End to End Learning for Self-Driving Cars", https://arxiv.org/abs/1604.07316
[2] Paul Heraty, "Behavioral Cloning Cheatsheet", https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb
