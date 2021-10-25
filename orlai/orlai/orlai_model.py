import numpy as np
import pandas as pd
import scipy.linalg as sc
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.model_selection import train_test_split


class Orlai:
    #TODO: make self.d, self.initialized private variables
    #TODO: make correct methods static


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

        self.w = compute_best_achievable_interpolator(X, y, c = self.c_e, prior_cov = np.eye(self.d), snr = snr_estimation, crossval_param = crossval_param)



    def predict(self, X):

        assert ((self.fitted) | (self.initialized)), 'need to train or initlaize the model first'

        assert X.shape[1] == len(self.w), 'data must have the same dimension as the training data (or initialization)'

        return X.dot(self.w)




def compute_best_achievable_interpolator(X, y, c, prior_cov, snr, crossval_param = 10):
    #TODO: parallelize this

    ''' If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosen as an estimate of the signal-to-noise ratio.'''

    c_inv = np.linalg.inv(c)
    d = X.shape[1]
    n = X.shape[0]

    if type(snr) == np.ndarray or type(snr) == list:

        # initialize dataframe where we save results
        df = pd.DataFrame([], columns = ['mu', 'error'])

        for mu in snr:

            error_crossvalidated = 0

            for j in range(crossval_param):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # random train test split

                n_train  = X_train.shape[0]
                n_test = X_test.shape[0]

                # calculate the best_achievable interpolator according to formula in paper
                auxi_matrix_train = np.linalg.inv(np.eye(n_train) + (mu/d)*X_train.dot(prior_cov.dot(X_train.T)))
                auxi_matrix_train_2 = ((mu/d)*prior_cov.dot(X_train.T) + (c_inv.dot(X_train.T)).dot( np.linalg.inv(X_train.dot(c_inv.dot(X_train.T))) ))
                w_e_train = auxi_matrix_train_2.dot(auxi_matrix_train.dot(y_train))

                y_test_pred = X_test.dot(w_e_train)

                error_crossvalidated += (np.linalg.norm(y_test - y_test_pred)**2)/n_test

            error_crossvalidated = error_crossvalidated/crossval_param

            df = df.append(pd.DataFrame(np.array([[mu, error_crossvalidated]]), columns = ['mu', 'error']))

        df = df.sort_values('error', ascending = True)

        snr = np.mean(df['mu'].iloc[:3].values)

    # calculate the best_achievable interpolator according to formula in paper
    auxi_matrix = np.linalg.inv(np.eye(n) + (snr/d)*X.dot(prior_cov.dot(X.T)))
    auxi_matrix_2 = ((snr/d)*prior_cov.dot(X.T) + (c_inv.dot(X.T)).dot( np.linalg.inv(X.dot(c_inv.dot(X.T))) ))
    w_e = auxi_matrix_2.dot(auxi_matrix.dot(y))

    return w_e



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