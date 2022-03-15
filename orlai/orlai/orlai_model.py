import numpy as np
import pandas as pd
import scipy.linalg as sc
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.model_selection import train_test_split
from orlai.utils.compute_orlai import compute_orlai


class Orlai:


    def __init__(self, init_w = None):

        self.fitted = False
        self.d = None
        self.c_e = None
        self.initialized = False

        if init_w is not None:
            self.initialized = True
            self.d = len(init_w)

        # initialize parameter
        self.w = init_w



    def fit(self, 
            X, 
            y, 
            empir = 'graphical_lasso', 
            alpha = 0.25, 
            prior_cov = None,
            snr_estimation = list(np.linspace(0.1,1,10))+list(np.linspace(1,10,10)),
            crossval_param = 10
            ):
        #TODO: implement crossvalidating alpha
        #TODO: implement giving data covariance matrix as input (for case when user has access to this knowledge)

        self.fitted = True
        self.d = X.shape[1]
        prior_cov = np.eye(self.d) if prior_cov is None else prior_cov

        assert len(y.shape) == 1, "output needs to be one-dimensional"
        assert X.shape[0] == len(y), "dimension of X and y don't match"
        assert prior_cov.shape[0] == prior_cov.shape[1] == self.d, "dimension of the prior covariance is not correct"

        # approximate the covariance matrix
        self.c_e = generate_c_empir(X, empir, alpha)

        self.w = compute_orlai(X, y, c = self.c_e, prior_cov = np.eye(self.d), snr = snr_estimation, crossval_param = crossval_param)



    def predict(self, X):

        assert ((self.fitted) | (self.initialized)), 'need to train or initlaize the model first'

        assert X.shape[1] == len(self.w), 'data must have the same dimension as the training data (or initialization)'

        return X.dot(self.w)



def generate_c_empir(X,empir, alpha = 0.25):

    if empir == 'basic':
        c_e = X.transpose().dot(X)/len(X)

    elif empir == 'ledoit_wolf':
        lw = LedoitWolf(assume_centered = True).fit(X)
        c_e = lw.covariance_

    elif empir == 'graphical_lasso':
        gl = GraphicalLasso(assume_centered = True, alpha = alpha, tol = 1e-4).fit(X)
        c_e = gl.covariance_

    else:
        raise AssertionError('specify regime of empirical approximation')

    return c_e
