# Optimal Response-Linear Achievable Interpolator

This repo contains the implementation of the optimal response-linear achievable interpolator from https://arxiv.org/abs/2110.11258 in python.

## Installation

Install the python package by cloning the repo and by `pip install orlai/` from the root of the directory. 

## Usage

### Generate a covariance matrix

```
import numpy as np
n = 300 # number of datapoints
n_test = int(n/5)
d = 400 # dimension
c = np.eye(d)

for i in range(d):
    for j in range(d):
            c[i,j] = 0.9**(abs(i-j))
```

### Generate features

```
X_train = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = n)
X_test = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = n_test)
```

### Generate data according to `y = X.dot(w_star) + xi`

```
w_star = np.random.multivariate_normal(mean = np.zeros(d),cov = np.eye(d))
y_train = X_train.dot(w_star) + np.random.multivariate_normal(mean = [0],cov = [[16]], size = n).reshape(n,)
y_test = X_test.dot(w_star) + np.random.multivariate_normal(mean = [0],cov = [[16]], size = n_test).reshape(n_test,)
```

### Fit and predict
```
from orlai.orlai_model import Orlai
orlai = Orlai()
orlai.fit(X_train, y_train)
y_test_orlai = orlai.predict(X_test)
```

### Fit and predict for minimum-norm-interpolator
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_test_reg = reg.predict(X_test)
```

### Compare

```
print(np.linalg.norm(y_test - y_test_orlai)**2/n_test)
print(np.linalg.norm(y_test - y_test_reg)**2/n_test)
```


