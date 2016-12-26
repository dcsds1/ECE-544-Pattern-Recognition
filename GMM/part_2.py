# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:19:26 2016

@author: changsongdong
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from PIL import Image

class GMM:
    """
    
    Cluster an image based on colors using EM algorithm.
    """
    
    def __init__(self, filename, n_clusters):
        
        # Read an image from a file as an array
        image = sp.misc.imread(filename)[:,:,0:3]
        self.height, self.weight, self.n_channel = image.shape
        self.n_sample = self.height * self.weight
        self.imarray = image.reshape(self.n_sample, self.n_channel)
        
        # Initialize parameters for Gaussian Mixture model
        # mean of Gaussian
        self.n_clusters = n_clusters
        self.init_mean = (np.random.rand(n_clusters, self.n_channel) + 0.5) * \
                         self.imarray.mean(0)
        
        # Covariance matrix of the distribution
        cov_i = np.eye(self.n_channel) * np.var(self.imarray, 0)
        cov=[]
        for i in xrange(n_clusters):
            cov.append(cov_i)
        self.init_cov = np.array(cov)
        
        # Weight of the Gaussian component
        self.init_omega = [1.0 / n_clusters] * n_clusters
                
    def gammaprob(self, x, omega, mean, cov):
        """
        
        Accepts the parameters of a Gaussian mixture model and the
        data samples as input, and outputs the posterior probability vectors
        
        gamma : n_sample by n_channel matrix
          the posterior probability matrix for 
        """
        
        #y = multivariate_normal.pdf(x, mean, cov)
        gamma = []
        for i in xrange(self.n_clusters):
            gamma.append(multivariate_normal.pdf(x, mean[i], cov[i]))
        gamma = omega * np.array(gamma).T
        row_sum = gamma.sum(axis=1)
        gamma /= row_sum[:, np.newaxis]
        return gamma
        
    def mstep(self, x, gamma):
        """
        computes the M-step for training a Gaussian mixture model
        
        parameters:
        
        x : n_sample by n_channel matrix
          input data
          
        omega : 1 by n_clusters
          possibility of each component
          
        mean : n_clusters by n_channel matrix
          mean of each component
          
        cov : (n_clusters * n_channel) by n_channel matrix
          covariance matrix for each component
        
        gamma : n_sample by n_channel matrix
          posterior probability
        """
        
        omega = (1.0 / self.n_sample) * np.sum(gamma, axis = 0)
        omega = [i / sum(omega) for i in omega]
        
        mean = np.dot(gamma.T, x) / np.sum(gamma, axis = 0)[:, None]
        
        cov = []
        cov_init = x - mean.reshape(self.n_clusters, 1, self.n_channel)
        for i in xrange(self.n_clusters):
            cov.append(np.dot(cov_init[i].T, gamma.T[i][:,None] * cov_init[i]))
        cov = np.array(cov)
        gamma_sum = np.sum(gamma, axis = 0)
        for i in xrange(self.n_clusters):
            cov[i] /= gamma_sum[i]
        
        return omega, mean, cov+np.eye(3)*0.1
        
    def train(self, iteration):
        """
        iteration : int
          training iterations
        """
        
        x = self.imarray
        omega = self.init_omega
        mean = self.init_mean
        cov = self.init_cov
        
        for i in xrange(iteration):
            gamma = self.gammaprob(x, omega, mean, cov)
            omega, mean, cov = self.mstep(x, gamma)
            
        label = np.argmax(gamma, axis=1)
        for i in xrange(self.n_sample):
            x[i] = mean[label[i]]
        img = Image.fromarray(x.reshape(self.height, self.weight,\
                                              self.n_channel))
        img.show()
        img.save('GMM_part2_result_10.bmp')
        
if __name__ == '__main__':
    gmm = GMM('corgi.png', 10)
#    imarray = gmm.imarray
#    mean = gmm.init_mean
#    cov = gmm.init_cov
#    omega = gmm.init_omega
    gmm.train(50)
