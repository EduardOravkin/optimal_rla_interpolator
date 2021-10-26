import numpy as np
import pandas as pd
import os
import multiprocessing
from functools import partial
from sklearn.model_selection import train_test_split


def compute_orlai_multi(X, y, c, prior_cov, snr, crossval_param = 10):
    #TODO: parallelize this

    ''' If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosen as an estimate of the signal-to-noise ratio.'''

    c_inv = np.linalg.inv(c)

    if type(snr) == np.ndarray or type(snr) == list:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = pool.map(partial(compute_crossvalidated_error_snr, 
                                        X=X, 
                                        y=y, 
                                        c_inv=c_inv, 
                                        prior_cov=prior_cov, 
                                        crossval_param=crossval_param, 
                                        test_size = 0.1), 
                                snr
                                ).sort()
        pool.close()
    
        snr = np.mean(results[-3:]) # estimate of the snr is the mean of the three best performing

    return orlai_formula(X, y, c_inv, prior_cov, snr)


def compute_crossvalidated_error_snr(snr, X, y, c_inv, prior_cov, crossval_param=10, test_size = 0.1):
    '''Given a particular snr, computes the average error on 
    {crossval_param} random subsets of size {test_size*len(y)} 
    of the data X, y.
    '''
    error_crossvalidated = 0

    for j in range(crossval_param):
        # random train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
        n_test = X_test.shape[0]

        # calculate the best_achievable interpolator according to formula in paper
        w_e_train = orlai_formula(X_train, y_train, c_inv, prior_cov, snr)

        # compute the test error
        y_test_pred = X_test.dot(w_e_train)
        error_crossvalidated += (np.linalg.norm(y_test - y_test_pred)**2)/n_test
        
    return error_crossvalidated/crossval_param


def orlai_formula(X, y, c_inv, prior_cov, snr):
    '''Computes the optimal response-linear achievable interpolator according 
    to the formula in Proposition 1 of the paper.
    '''
    n, d  = X.shape

    assert d == c_inv.shape[0] == c_inv.shape[1], 'Incorrect dimension of the data covariance matrix.'
    assert d == prior_cov.shape[0] == prior_cov.shape[1], 'Incorrect dimension of the data covariance matrix.'
    assert n == len(y), "Incorrect dimension of the response variable."
    assert snr > 0, 'The signal-to-noise ratio must be positive.'

    auxi_matrix = np.linalg.inv(np.eye(n) + (snr/d)*X.dot(prior_cov.dot(X.T)))
    auxi_matrix_2 = ((snr/d)*prior_cov.dot(X.T) + (c_inv.dot(X.T)).dot( np.linalg.inv(X.dot(c_inv.dot(X.T))) ))
    return auxi_matrix_2.dot(auxi_matrix.dot(y))


def compute_orlai(X, y, c, prior_cov, snr, crossval_param = 10):
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