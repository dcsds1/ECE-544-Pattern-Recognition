# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:16:02 2016

@author: changsongdong
"""

import numpy as np
import random
import math
import copy

lr = 0.0001          # learning rate
batchsize = 44     # mini-batch size
epoch = 2000       # training epochs
#display = 0.01       # display after every ten epochs
number_of_hidden_neurons = 50

def get_data(set_type):
    """Get data from files and storage them in an array. 
    
    set_type    the type of data set you want to build, 
                including train dataset, dev dataset and eval dataset
    """
    
    data_path = {'train': 'train/lab/hw2train_labels.txt', 
                 'dev': 'dev/lab/hw2dev_labels.txt',                  
                 'eval': 'eval/lab/hw2eval_labels.txt'} 
            
    #load the label file contents into a array
    label_array = np.loadtxt(data_path[set_type], dtype='string')     
    
    labelset = [int(i) for i, j in label_array]
    j = 0
    total_dataset = []
    
    #build the data set and label set
    for i in range(len(label_array)):
        with open(label_array[i][1]) as data_file:
            data = data_file.readlines()
            if len(data) < 70:
                # delete label from label_set
                del labelset[j]
                continue
            else:
                dataset = []
                for i in np.arange(70):
                    dataset.extend(data[i].split())

                total_dataset.append(dataset)                
                j += 1

    #labelset = np.asarray(labelset)
    label_set = np.zeros([len(labelset), 9])
    for i in range(len(label_set)):
        label_set[i][labelset[i]] = 1
        
    data_set = np.zeros([len(total_dataset), 16 * 70])
    for i in range(len(data_set)):
        data_set[i] = total_dataset[i]

    return data_set, label_set #return the shuffled data set and label set

#train_data =  np.loadtxt("train_data")
#train_label =  np.loadtxt("train_label")
#test_data =  np.loadtxt("test_data")
#test_label =  np.loadtxt("test_label")
train_data, train_label = get_data('train')
test_data, test_label = get_data('eval')  

num_examples = len(train_data) # number of total training samples

# ================= Nonlinearities ===================== 
class Sigmoid:
    def forward(self, x):
        return 1. / (1 + np.exp(-x))
    def backward(self, y):
        return (1. - y) * y
class Tanh:    
    def forward(self, x):
        return np.tanh(x)
    def backward(self, y):
        return ((1. - np.square(y)))

class Relu:
    def forward(self,x):
        return x * (x > 0)
    def backward(self, y):
        return 1. * (y > 0)

# ================== accuracy and loss ===================
def compute_acc(data, label, w_1, w_2, w_3, b_1, b_2, b_3, sigmoid):
    """Compute the accuracy
    
    """
    acc = 0
    for i in range(len(label)):
        # forward propagation
        # first hidden layer
        a_1 = np.dot(w_1, data[i].transpose()) + b_1
        z_1 = sigmoid.forward(a_1)
        
        # second hidden layer
        a_2 = np.dot(w_2, z_1) + b_2
        z_2 = sigmoid.forward(a_2)
        
        # output
        a_out = np.dot(w_3, z_2) + b_3
        
        # softmax
        y_hat = np.exp(a_out) / np.sum(np.exp(a_out))
    
        #if label[i] == round(np.dot(w, data[i])+b):
        if label[i][np.argmax(y_hat)] == 1:
            acc += 1

    return acc / float(len(label))
 
# ================= Forward Propagation ===================
class Layer:
    def __init__(self, number_of_neurons, length_of_input):
        self.weight = 0.2 * np.random.randn(number_of_neurons, length_of_input)
        self.b = 0
        
def train(train_data, train_label, epoch):
    sigmoid = Sigmoid()
    tanh = Tanh()
    relu = Relu()
    
    h1 = Layer(number_of_hidden_neurons, 70 * 16)
    h2 = Layer(number_of_hidden_neurons, number_of_hidden_neurons)
    h3 = Layer(9, number_of_hidden_neurons)
    
    error = 0
    w_1 = h1.weight
    w_2 = h2.weight
    w_3 = h3.weight
    b_1 = h1.b
    b_2 = h2.b
    b_3 = h3.b
    
    for i in range(epoch):
        # shuffle data in each epoch
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_data = train_data[perm]
        train_label = train_label[perm]
   
        for start in np.arange(0, num_examples, batchsize):
            # mini-batch gradient descent            
            data_batch = train_data[start:start+batchsize]
            label_batch = train_label[start:start+batchsize]
            
            delta_o = 0 
            dao_w3 = 0
            dw_3 = np.zeros(w_3.shape)
            
            # derivatives of 2nd hiddenlayer
            da_w2 = 0
            dz2_a2 = 0
            delta_2 = 0 
            dw_2 = np.zeros(w_2.shape)
            
            # derivatives of 1st hidden layer
            dz1_a1 = 0
            da1_w1 = 0
            delta_1 = 0
            dw_1 = np.zeros(w_1.shape)
            
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
                error = -np.dot(label_batch[j], np.log(y_hat))
            
                # back propagation
                # derivatives of output
                delta_o = y_hat - label_batch[j].transpose() # merge derivative of error WRT a_out
                dao_w3 = z_2 # derivative of a_out WRT w_3
                dw_3 += np.outer(delta_o, dao_w3)
                
                # derivatives of 2nd hiddenlayer
                da_w2 = z_1
                dz2_a2 = sigmoid.backward(z_2) # derivative of z_2 WRT a_2
                delta_2 = dz2_a2 * np.dot(w_3.transpose(), delta_o)
                dw_2 += np.outer(delta_2, da_w2)
                
                # derivatives of 1st hidden layer
                da1_w1 = data_batch[j] # derivative of a_1 WRT w_1
                dz1_a1 = sigmoid.backward(z_1) # derivative of z_1 WRT a_1
                delta_1 = dz1_a1 * np.dot(w_2.transpose(), delta_2)
                dw_1 += np.outer(delta_1, da1_w1)
                
            # back propagation
#            xx_3 = 1/batchsize * lr * dw_3 # gradient descent of output
#            xx_2 = 1/batchsize * lr * dw_2 # gradient descent of 2nd hidden layer
#            xx_1 = 1/batchsize * lr * dw_1 # gradient descent of 1st hidden layer
            
            w_3 -= lr * dw_3 # gradient descent of output
            w_2 -= lr * dw_2 # gradient descent of 2nd hidden layer
            w_1 -= lr * dw_1 # gradient descent of 1st hidden layer
        acc = compute_acc(train_data, train_label, w_1, w_2, w_3, b_1, b_2, b_3, sigmoid)
        test_acc = compute_acc(test_data, test_label, w_1, w_2, w_3, b_1, b_2, b_3, sigmoid)
        print 'epoch %s : train_accuracy is %f., test_accuracy is %f' % (i, acc, test_acc)
    return error

    
  
print train(train_data, train_label, epoch)