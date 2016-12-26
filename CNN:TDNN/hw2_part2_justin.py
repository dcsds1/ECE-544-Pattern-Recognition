# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:16:02 2016

@author: changsongdong
"""

import numpy as np
import random
import math

lr = 0.01          # learning rate
batchsize = 512     # mini-batch size
epoch = 500         # training epochs
display = 10        # display after every ten epochs
num_examples = 5060 # number of total training samples

number_of_hidden_neurons = 10

# ================= Nonlinearities ======================

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + math.exp(-x))
    def backward(self, x):
        output = self.forward(x)
        return (1 - output) * output
class Tanh:    
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x):
        output = self.forward(x)
        return ((1 - np.square(output)))

class Relu:
    def forward(self,x):
        return x * (x > 0)
    def backward(self, x):
        if x > 0:
            return 1
        else:
            return 0
    

# ================= Forward Propagation ===================
class Layer:
    def __init__(self, number_of_neurons, length_of_input):
        self.weight = np.random.randn(number_of_neurons, length_of_input)
        self.b = np.zeros(number_of_neurons)
        
def train(train_data, train_label, epoch):
    sigmoid = Sigmoid()
    tanh = Tanh()
    relu = Relu()
    num_nueral = 10
    
    h1 = Layer(num_nueral, 70 * 16)
    h2 = Layer(num_nueral, num_nueral)
    h3 = Layer(num_nueral - 1, num_nueral)
    
    batchsize = 110
    data_batch = np.zeros([batchsize, 1120])
    
    error = 0
    w_1 = h1.weight
    w_2 = h2.weight
    w_3 = h3.weight
    dy_a, de_s = 0
    
    for i in range(epoch):
        # shuffle data in each epoch
        perm = np.arange(len(train_data))
        np.random.shuffle(perm)
        train_data = train_data[perm]
        train_label = train_label[perm]           
        
        # mini-batch gradient descent
        for start in np.arange(0, 5060, batchsize):
            data_batch = train_data[start : start + batchsize]
            label_batch = train_label[start : start + batchsize]
            for i in range(batchsize):
                # first hidden layer
                b_1 = h1.b
                a_1 = np.dot(data_batch[i], w_1.transpose()) + b_1
                z_1 = sigmoid.forward(a_1)
            
                # second hidden layer
                b_2 = h2.b
                a_2 = np.dot(z_1, w_2.transpose()) + b_2
                z_2 = sigmoid.forward(a_2)
            
                # output
                b_3 = h3.b
                a_out = np.dot(z_2, w_3.transpose()) + b_3
                y_hat = sigmoid.forward(a_o)
            
                # softmax
                y_hat = np.exp(y_hat) / sum(np.exp(y_hat))
        
                # negative log likelihood loss
                error += sum(np.log(y_hat))
                
                # derivatives of output
#                de_s += sum(label_batch[i] / y_hat, axis=1) # derivative of error WRT softmax
#                ds_y +=  # derivative of softmax WRT y_hat
                #dy_ao += sigmoid.backward(a_o) # derivative of y_hat WRT a_out
                dao_z2 +=  # derivative of a_out WRT z_2
                
                # derivatives of 2nd layer
                dz2_a2 += sigmoid.backward(a_2) # derivative of z_2 WRT a_2
                da2_z1 +=  # derivative of a_2 WRT z_1
                
                # derivatives of 1st layer
                dz1_a1 += sigmoid.backward(a_1) # derivative of z_1 WRT a_1
                da1_databatch += # derivative of a_1 WRT data_batch[i]
                
            # back propagation
            w_3 -= de_s * ds_y * dy_ao * da_z2 # gradient descent of output
            w_2 -= dz2_a2 * da2_z1 # gradient descent of 2nd hidden layer
            w_1 -= dz1_a1 * da1_databatch # gradient descent of 1st hidden layer

        
        
#==========
for j in range(batchsize):
                
                # forward propagation
                # first hidden layer
                a_1 = np.dot(w_1, data_batch[j].transpose()) + b_1  
                z_1 = sigmoid.forward(a_1)
                
                # second hidden layer
                a_2 = np.dot(w_2, z_1) + b_2
                z_2 = sigmoid.forward(a_2)
                
                # output
                a_out = np.dot(w_3, z_2) + b_3
                
                # softmax
                y_hat = np.exp(a_out) / np.sum(np.exp(a_out))
            
                # negative log likelihood loss
                error += -np.dot(label_batch[j], np.log(y_hat))
            
                # back propagation
                # derivatives of output
                delta_o += y_hat - label_batch[0].transpose()
                da_w += z_2
                
                # derivatives of 2nd hiddenlayer
                da_w2 += z_1
                dz2_a2 += sigmoid.backward(z_2) # derivative of z_2 WRT a_2
                delta_2 += dz2_a2 * w_3 * delta_o
                
                # derivatives of 1st hidden layer
                dz1_a1 += sigmoid.backward(a_1) # derivative of z_1 WRT a_1
                da1_w1 += data_batch[i] # derivative of a_1 WRT w_1
                delta_1 += dz1_a1 * w_2 * delta_2

            # back propagation
            w_3 -= 1/batchsize * lr * delta_o * da_w.transpose() # gradient descent of output
            w_2 -= 1/batchsize * lr * delta_2 * da_w2 # gradient descent of 2nd hidden layer
            w_1 -= 1/batchsize * lr * delta_1 * da1_w1 # gradient descent of 1st hidden layer
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        