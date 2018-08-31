## Udacity - Self Driving Car Nanodegree (Term 3) - Semantic Segmentation Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Background
---
In this project, our goal is to implement a Fully Convolutional Network (FCN) that will perform semantic segmentation on road images to label the pixels of a road in a different color.

Overview of Repository
---
This repository contains the following source code file that I have forked from the [main repository](https://github.com/udacity/CarND-Semantic-Segmentation) and subsequently modified for this project:

1.  [main.py](https://github.com/MartinKan/CarND-Semantic-Segmentation/blob/master/src/main.py)

Summary of the Model
---
The main.py class implements the [FCN-8 model](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) to perform semantic segmentation on road images.  The FCN-8 model contains two parts: an encoder portion and a decoder portion.  The encoder portion downsamples the input image to smaller dimensions so that it will be computationally efficient to process. While a decoder portion upsamples the output of the encoder and restores the processed image to the original image size so that segmantation on the original image can be done.  

For the project, the encoder portion of the model is implemented using a pretrained VGG Model that is imported into the code (lines 21-46), the relevant layers are then extracted to do the required transposed convolutions and skip connections during upsampling (lines 51-115).  Other common features of neural network such as batch processing (to reduce memory footprint of each training step) and softmax cross entropy (to compute loss) are also implemented.

Results
---
Here are some of the sample results from the model using the road images from the Kitti Road Dataset:

![alt text](https://github.com/MartinKan/CarND-Semantic-Segmentation/blob/master/src/results/um_000002.png)
![alt text](https://github.com/MartinKan/CarND-Semantic-Segmentation/blob/master/src/results/um_000063.png)
![alt text](https://github.com/MartinKan/CarND-Semantic-Segmentation/blob/master/src/results/um_000069.png)
![alt text](https://github.com/MartinKan/CarND-Semantic-Segmentation/blob/master/src/results/um_000083.png)
![alt text](https://github.com/MartinKan/CarND-Semantic-Segmentation/blob/master/src/results/um_000091.png)

Running the code
---

To run the code, use the following command:

	python main.py

This will train the network and output the processed road images into a separate /runs folder.