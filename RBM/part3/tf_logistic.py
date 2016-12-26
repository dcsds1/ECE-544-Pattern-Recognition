# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:25:23 2016

@author: changsongdong
"""

#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

class Logistic:
    def __init__(self, num_features, num_class, learning_rate):
       
        self.x = tf.placeholder(tf.float32, [None, num_features])
        
        self.W = tf.Variable(tf.zeros([num_features, num_class]))
        self.b = tf.Variable(tf.zeros([num_class]))
        
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        
        self.y_ = tf.placeholder(tf.float32, [None, num_class])

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
        
        init = tf.initialize_all_variables()
        
        self.sess = tf.Session()
        self.sess.run(init)

    def fit(self, train_data, train_label, test_data, test_label, batchsize):
        num_examples, features = train_data.shape
        for i in range(10):
            for start in np.arange(0, num_examples, batchsize):
                data_batch = train_data[start:start+batchsize]
                label_batch = train_label[start:start+batchsize]
                self.sess.run(self.train_step, feed_dict={self.x: data_batch, 
                                                          self.y_: label_batch})
              
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print 'logistic train accuracy is: %f' % self.sess.run(accuracy, 
                                                         feed_dict={self.x: train_data, 
                                                                    self.y_: train_label})
            print 'logistic test accuracy is: %f' % self.sess.run(accuracy, 
                                                         feed_dict={self.x: test_data, 
                                                                    self.y_: test_label})
    def plot_matrix(self, train_data, train_label, test_data, test_label):  
        cm = tf.contrib.metrics.confusion_matrix(tf.argmax(self.y, 1), tf.argmax(self.y_,1))
        cm1 = self.sess.run(cm, feed_dict={self.x: train_data, self.y_: train_label})
        cm2= self.sess.run(cm, feed_dict={self.x: test_data, self.y_: test_label})
        
        index = ['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
        #np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cm1, index, title='Confusion matrix for Training Data')
        plt.show()
        
        plt.figure()
        plot_confusion_matrix(cm2, index, title='Confusion matrix for Testing Data')
        plt.show()     
        
#if __name__ == '__main__':
#    logistic = Logistic(784, 10, 0.5)
#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels,\
#                                                     mnist.test.images, mnist.test.labels
#    logistic.fit(train_data, train_label, test_data, test_label, 1000)
#    logistic.plot_matrix(train_data, train_label, test_data, test_label)