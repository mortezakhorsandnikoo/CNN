#!/usr/bin/env python
# coding: utf-8


#libraries
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


#import image 
pil_im = Image.open("D:/AI_Machinlearning/datasets/animal/cat_dog/Cat/10.jpg")

#convert image to greyscale mode
grey_img= pil_im.convert(mode="L")
grey_img


#conver jpg to matrix
img_matrix=np.asarray(gray_img)

print(img_matrix.shape)

img_matrix



#mean
mean=img_matrix.mean()
mean


#standard deviation
std=img_matrix.std()
std



#horizontal
kernel=np.array([[-1,-2,-1],
                  [0 ,0, 0],
                  [1, 2,1]])
                 



#first Convolution
conv=convolve2d(img_matrix , kernel ,boundary='symm', mode='same')

conv.shape


plt.imshow(conv ,alpha=0.9 ,  cmap="bone")   #https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.show()



#vertical
kernel2=np.array([[-1,0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])

#second Convolution
conv2=convolve2d(conv , kernel2 ,boundary='symm', mode='same')




plt.imshow(conv2 , interpolation='mitchell', cmap="bone" , alpha=0.9)
plt.show()




#diagonal
kernel3=np.array([[-3,5,5],
                 [-3,0,5],
                 [-3,-3,-3]])



#third Convolution
conv3=convolve2d(conv2 , kernel3 ,boundary='symm', mode='same')



plt.imshow(conv3 , interpolation='mitchell', cmap="bone" , alpha=0.9)
plt.show()



#concatenate 3 metrices
new_layer=np.dstack((conv , conv2 , conv3))



new_layer.shape



maxValue = np.amax(new_layer)
minValue = np.amin(new_layer)

print( maxValue , minValue , new_layer.mean())




Image = np.clip(new_layer, 0, 255)

plt.imshow(Image)

