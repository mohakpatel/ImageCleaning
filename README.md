# ImageCleaning

In this project I aimed at cleaning noisy microscope images. 


My motivation behind this project is that I write image processing algorithms to extract meaningful quantitative data out of micoscope images. However, writing robust algorithms which work in presense of noise is often challenging. My task becomes easier if I don't have the optical distortions and noise in the images. 

Even for people not performing quantiative analysis with images, getting clean images can be benefical. Small features can be observed in images.
 if we can performe deconvolution and noise removal. While there are many algorithms for deconvolution, performing deconvolution in presence of noise is very challenging. 

In this project, I use convolutional auto-encoders to performe image deconvolution and remove noise from the images at the same time. To train the model, I synthetically generate my training images based on the optical properties of an microscope. 

Here is a sneak-peak of the on the performance of my deep learning model:

![](https://github.com/mohakpatel/ImageCleaning/blob/master/data/Results.png)


## Files

* [ImageCleaning.ipynb](https://github.com/mohakpatel/ImageCleaning/blob/master/src/ImageCleaning.ipynb): I describe the image generation process, and the training, testing and visualization of the model results. 
* [model.py](https://github.com/mohakpatel/ImageCleaning/blob/master/src/model.py): Convolutional auto-encoder model to deconvolve and clean the images implemented in Tensorflow
* [image_generation.py](https://github.com/mohakpatel/ImageCleaning/blob/master/src/image_generation.py): Synthetic microscope image generation
* [imshow3D.py](https://github.com/mohakpatel/ImageCleaning/blob/master/src/imshow3D.py): 3D image slice viewer developed using ipywidgets


## Requirements

I used the following version of libraries in Python 3 for my model
* Tensorflow 1.12
* Numpy 1.15.1
* Scipy 1.1.0
* scikit-learn 0.19.2
* matplotlib 2.2.3
* ipywidgets 7.4.1
* time
* os





