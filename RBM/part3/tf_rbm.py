# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3  2016

@author: changsongdong
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from utils import tile_raster_images

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=1.0, batchsize=100):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        
        self.X = tf.placeholder(tf.float32, [None, self.n_visible])

        self.w = tf.placeholder(tf.float32, [self.n_visible, self.n_hidden])
        self.visible_bias = tf.placeholder(tf.float32, [self.n_visible])
        self.hidden_bias = tf.placeholder(tf.float32, [self.n_hidden])
        
        self.n_w = np.zeros([n_visible, n_hidden], np.float32)
        self.n_visible_bias = np.zeros([n_visible], np.float32)
        self.n_hidden_bias = np.zeros([n_hidden], np.float32)
        
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
    def sample_bernoulli(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))
        
    def transform(self, data):
        """
        data: mnist (n_samples, 784)
        
        return: (n_samples, n_hidden)
        """
        
        compute_hidden = tf.nn.sigmoid(tf.matmul(data, self.n_w) + self.n_hidden_bias)
        return self.sess.run(compute_hidden, feed_dict={self.X: data})
        
    def fit(self, train_X, test_X):
        h0 = self.sample_bernoulli(tf.nn.sigmoid(tf.matmul(self.X, self.w) + self.hidden_bias))
        v1 = self.sample_bernoulli(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(self.w)) + self.visible_bias))
        h1 = tf.nn.sigmoid(tf.matmul(v1,self.w) + self.hidden_bias)
        
        w_positive_grad = tf.matmul(tf.transpose(self.X), h0)
        w_negative_grad = tf.matmul(tf.transpose(v1), h1)
        
        update_w = self.w + self.learning_rate * \
                   (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(self.X)[0])
        update_visible_bias = self.visible_bias + self.learning_rate * tf.reduce_mean(self.X - v1, 0)
        update_hidden_bias = self.hidden_bias + self.learning_rate * tf.reduce_mean(h0 - h1, 0)
        
        h_sample = self.sample_bernoulli(tf.nn.sigmoid(tf.matmul(self.X, self.w) + self.hidden_bias))
        v_sample = self.sample_bernoulli(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(self.w)) + self.visible_bias))
        err = self.X - v_sample
        acc = 1 - tf.reduce_mean(err * err)
        
        for start, end in zip(range(0, len(train_X), self.batchsize), 
                              range(self.batchsize, len(train_X), self.batchsize)):
            batch = train_X[start:end]
            self.n_w = self.sess.run(update_w, feed_dict={
                           self.X: batch, 
                           self.w: self.n_w, 
                           self.visible_bias: self.n_visible_bias, 
                           self.hidden_bias: self.n_hidden_bias})
            self.n_visible_bias = self.sess.run(update_visible_bias, feed_dict={
                            self.X: batch, 
                            self.w: self.n_w, 
                            self.visible_bias: self.n_visible_bias, 
                            self.hidden_bias: self.n_hidden_bias})
            self.n_hidden_bias = self.sess.run(update_hidden_bias, feed_dict={
                            self.X: batch, 
                            self.w: self.n_w, 
                            self.visible_bias: self.n_visible_bias, 
                            self.hidden_bias: self.n_hidden_bias})
                            
            if start % 10000 == 0:
                print 'rbm training accuracy: %f' % self.sess.run(acc, 
                                                             feed_dict={
                                                             self.X: train_X, 
                                                             self.w: self.n_w, 
                                                             self.visible_bias: self.n_visible_bias, 
                                                             self.hidden_bias: self.n_hidden_bias})
                print 'rbm testing accuracy: %f' % self.sess.run(acc, 
                                                             feed_dict={
                                                             self.X: test_X, 
                                                             self.w: self.n_w, 
                                                             self.visible_bias: self.n_visible_bias, 
                                                             self.hidden_bias: self.n_hidden_bias})
                                                             
#        return self.transform(train_X), self.transform(test_X)
    
                image = Image.fromarray(
                    tile_raster_images(
                        X=self.n_w.T,
                        img_shape=(28, 28),
                        tile_shape=(8, 8),
                        tile_spacing=(1, 1)
                    )
                )
                image.save("rbm_%d.png" % (start / 10000))

if __name__ == '__main__':
    rbm = RBM(n_visible=784, n_hidden=200, learning_rate=1.0, batchsize=100)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels,\
                                                     mnist.test.images, mnist.test.labels

    rbm.fit(train_data, test_data)
