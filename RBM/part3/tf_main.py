# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:19:39 2016

@author: changsongdong
"""

from tf_logistic import Logistic
from tf_rbm import RBM
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA

def logistic():
    logistic = Logistic(784, 10, 0.5)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels,\
                                                     mnist.test.images, mnist.test.labels

    logistic.fit(train_data, train_label, test_data, test_label, 1000)

def rbm_logistic():
    rbm = RBM(784, 200, learning_rate=1.0, batchsize=100)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels,\
                                                     mnist.test.images, mnist.test.labels

    reduced_train_data, reduced_test_data = rbm.fit(train_data, test_data)
    logistic = Logistic(200, 10, 0.5)
    logistic.fit(reduced_train_data, train_label, reduced_test_data, test_label, 1000)
    logistic.plot_matrix(reduced_train_data, train_label, reduced_test_data, test_label)
    
def pca_logistic():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels,\
                                                     mnist.test.images, mnist.test.labels

    pca=PCA(n_components=200)  
    reduced_train_data=pca.fit_transform(train_data)
    reduced_test_data=pca.transform(test_data)
    logistic = Logistic(200, 10, 0.5)
    logistic.fit(reduced_train_data, train_label, reduced_test_data, test_label, 1000)
    logistic.plot_matrix(reduced_train_data, train_label, reduced_test_data, test_label)
    
def srbm_logistic():
    rbm_1 = RBM(784, 500, learning_rate=1.0, batchsize=100)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels,\
                                                     mnist.test.images, mnist.test.labels
    reduced_train_data, reduced_test_data = rbm_1.fit(train_data, test_data)
    
    rbm_2 = RBM(500, 200, learning_rate=1.0, batchsize=100)
    reduced_train_data, reduced_test_data = rbm_2.fit(reduced_train_data, reduced_test_data)
    
    logistic = Logistic(200, 10, 5)
    logistic.fit(reduced_train_data, train_label, reduced_test_data, test_label, 1000)
    logistic.plot_matrix(reduced_train_data, train_label, reduced_test_data, test_label)

#logistic() 
#logistic train accuracy is: 0.930855
#logistic test accuracy is: 0.925700

#rbm_logistic() 
#logistic train accuracy is: 0.942418
#logistic test accuracy is: 0.945400

#pca_logistic() 
#logistic train accuracy is: 0.926418
#logistic test accuracy is: 0.924600

srbm_logistic() 
#lr=0.1 logistic train accuracy is: 0.915036 logistic test accuracy is: 0.920200
#lr=1 logistic train accuracy is: 0.928382 logistic test accuracy is: 0.930600
#lr=1.5 logistic train accuracy is: 0.930073 logistic test accuracy is: 0.930600
#lr=2 logistic train accuracy is: 0.931145 logistic test accuracy is: 0.930200
#lr=3 logistic train accuracy is: 0.931709 logistic test accuracy is: 0.931600
#lr=4 logistic train accuracy is: 0.932509 logistic test accuracy is: 0.932400
#lr=5 logistic train accuracy is: 0.932982 logistic test accuracy is: 0.932400
#lr=6 logistic train accuracy is: 0.932491 logistic test accuracy is: 0.932600
#lr=9 logistic train accuracy is: 0.931564 logistic test accuracy is: 0.932000

