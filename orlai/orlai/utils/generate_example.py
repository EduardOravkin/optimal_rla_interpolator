import numpy as np


def generate_example():
    ''' Utility function which generates an example linear regression problem
    and returns all of its parameters.
    '''
    
    # number of datapoints and parameters
    n = 500 
    d = 1000 

    # generate covariance matrix of data and prior
    c = np.eye(d)
    for i in range(d):
        for j in range(d):
                c[i,j] = 0.5**(abs(i-j))
    prior_cov = np.eye(d)

    # generate train and test data
    X_train = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = n)
    X_test = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = int(n/10))

    # generate true parameter and response vector
    w_star = np.random.multivariate_normal(mean = np.zeros(d),cov = prior_cov)
    y_train = X_train.dot(w_star) + np.random.multivariate_normal(mean = [0],cov = [[1]], size = n).reshape(n,)
    y_test = X_test.dot(w_star) + np.random.multivariate_normal(mean = [0],cov = [[1]], size = int(n/10)).reshape(int(n/10),)   

    return n, d, c, prior_cov, X_train, X_test, y_train, y_test, w_star