# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:38:13 2016

@author: changsongdong
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

lr = 0.1          # learning rate
batchsize = 506     # mini-batch size
epoch = 3000         # training epochs
display = 50        # display after every ten epochs

# sigmoid:
# lr = 0.01, batchsize = 506, epoch = 2000, train acc = 0.411067, test acc=0.418605
# lr = 0.01, batchsize = 506, epoch = 3000, train acc = 0.467391, test acc=0.433653

# lr = 0.1, batchsize = 506, epoch = 2000, train acc = 0.405731, test acc=0.435021
# lr = 0.1, batchsize = 506, epoch = 3000, train acc = 0.562055, test acc=0.514364

# add 3th conv layer:lr = 0.1, batchsize = 506, epoch = 3000, train acc = 0.678261, test acc = 0.622435
# add 4th conv layer:lr = 0.1, batchsize = 506, epoch = 3000, train acc = 0.731028，test acc =0.64827
# first hidden layer kernelsize 5*1 train acc = 0.528063, test acc = 0.525308
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

    return data_set, label_set, labelset #return the shuffled data set and label set

   
train_data, train_label, trainlabelset = get_data('train')
test_data, test_label, testlabelset = get_data('dev')

#train_data =  np.loadtxt("train_data")
#train_label =  np.loadtxt("train_label")
#test_data =  np.loadtxt("test_data")
#test_label =  np.loadtxt("test_label")

num_examples = len(train_data) # number of total training samples

# dataset x and label set y
x = tf.placeholder(tf.float32, [None, 1120])
y = tf.placeholder(tf.float32, [None, 9])
t = tf.Variable(testlabelset)
    
# weight initialization function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# layer definition
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

x_input = tf.reshape(x, shape=[-1, 70, 16, 1]) # reshape input

# first hidden layer parameters initialization
W_conv1 = weight_variable([5, 16, 1, 8])
b_conv1 = bias_variable([8])
o_conv1 = tf.nn.sigmoid(conv2d(x_input, W_conv1) + b_conv1)

# second hidden layer parameters initialization
W_conv2 = weight_variable([5, 1, 8, 9])
b_conv2 = bias_variable([9])
o_conv2 = tf.nn.sigmoid(conv2d(o_conv1, W_conv2) + b_conv2)

# multi-class logistic regression (softmax)
a_output = tf.reduce_sum(o_conv2, 1)
a_flat = tf.reshape(a_output, [-1, 9])

y_hat=tf.nn.softmax(a_flat)

#
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), 
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(epoch):
    # shuffle data in each epoch
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    train_data = train_data[perm]
    train_label = train_label[perm]

    if i % display == 0:
        train_accuracy = accuracy.eval(feed_dict={x: train_data, 
                                                  y: train_label})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        
    # mini-batch gradient descent
    for start in np.arange(0, num_examples, batchsize):
        data_batch = train_data[start:start+batchsize]
        label_batch = train_label[start:start+batchsize]
        train_step.run(feed_dict={x: data_batch, 
                                  y: label_batch})

print("test accuracy %g"%accuracy.eval(feed_dict={x: test_data, 
                                                  y: test_label}))
                                                  
# ============== draw confusion matrix ==============
prediction=tf.argmax(y,1)
y = prediction.eval(feed_dict={y: test_label})

prediction=tf.argmax(y_hat,1)
y_pred = prediction.eval(feed_dict={x: test_data})

matrix = confusion_matrix(y, y_pred)
index = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, index, title='Confusion Matrix')
plt.show()