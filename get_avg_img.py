#!/usr/bin/env python
# coding: utf-8

# In[1]:


# source: https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil

import os, numpy, PIL
from PIL import Image

# Access all PNG files in directory
#allfiles=os.listdir(os.getcwd())
imgDir = "Data/images/alcoholic/"
imageName = "AverageAlcoholic.png"
allfiles=os.listdir(imgDir)
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]
len(imlist)



# In[2]:


# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(imgDir + imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imarr=numpy.array(Image.open(imgDir + im),dtype=numpy.float)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save(imageName)
#out.show()


# In[ ]:




