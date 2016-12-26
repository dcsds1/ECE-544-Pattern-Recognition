# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:42:31 2016

@author: justinliu
"""

import os
import struct
import numpy as np
from numpy import random as rng
from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
from utils import tile_raster_images
import timeit

def read(dataset = "training", path = "."):
    """
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img
# In[]    
def data_reshape(images):
    """
    """
    
    reshaped_images = np.reshape(images, (len(images), 784))
    for i in range(len(images)):
        for j in range(784):
            if (reshaped_images[i][j] != 0):
                reshaped_images[i][j] = 1
    
    return reshaped_images
# In[]
def sigmoid(x):
    """
    """
    return 1. / (1. + np.exp(-x))

# In[]    
def training(images, epochs, k, momentum, lr):
    """
    """
    
    n = 200 # number of hidden units
    m = 784 # number of visible units
    W = rng.randn(m, n)
    c = np.zeros(n)
    b = np.zeros(m)
    reconstruction_error = 0
    
    for epoch in xrange(epochs):
        print 'epoch: ', epoch
        ph_mean, ph_sample = sample_h_given_v(images, W, c)
        chain_start = ph_sample
    
        for step in xrange(k):
            if step == 0:
                nv_mean, nv_sample = sample_v_given_h(chain_start, W, b)
                nh_mean, nh_sample = sample_h_given_v(nv_sample, W, c)
            else:
                nv_mean, nv_sample = sample_v_given_h(nh_sample, W, b)
                nh_mean, nh_sample = sample_h_given_v(nv_sample, W, c)
                
        W = momentum * W + lr * (np.dot(images.T, ph_mean) - np.dot(nv_sample.T, nh_mean))
        b = momentum * b + lr * np.mean(images - nv_sample, axis=0)
        c = momentum * c + lr * np.mean(ph_mean - nh_mean, axis=0)
        
        nv_mean, nv_sample = sample_v_given_h(chain_start, W, b)
        reconstruction_error = np.linalg.norm(images - nv_mean)
        print 'Reconstruction_error: ', reconstruction_error
        
    return W
    
# In[]
def sample_h_given_v(v0_sample, W, c):
    h1_mean = sigmoid(np.dot(v0_sample, W) + c)
    h1_sample = rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)

    return h1_mean, h1_sample

def sample_v_given_h(h0_sample, W, b):
    v1_mean = sigmoid(np.dot(h0_sample, W.T) + b)
    v1_sample = rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)   
   
    return v1_mean, v1_sample

# In[]
raw_images= read()
images = data_reshape(raw_images)
#rng=np.random.RandomState(123)
#rbm = RBM(input=images, n_visible=784, n_hidden=200, rng=rng)
#for epoch in xrange(10):
#        rbm.contrastive_divergence(lr=0.1, k=1)
#w=rbm.W
#w=np.reshape(w.transpose(), (200, 28, 28))
#print w.shape
start_time = timeit.default_timer()
w = training(images, 500, 1, 0.9, 0.001)
w_ori=np.reshape(w.transpose(), (200, 28, 28))
w=-w_ori

image = Image.fromarray(tile_raster_images(w[100:164], img_shape=(28, 28), 
                       tile_shape=(8,8), tile_spacing=(1, 1),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True))
end_time = timeit.default_timer()
image.save("filters.png")
plt.imshow(w[0], interpolation='nearest', cmap=cm.gray)
plt.show()
print ('Training took %f minutes' % ((end_time-start_time) / 60.))