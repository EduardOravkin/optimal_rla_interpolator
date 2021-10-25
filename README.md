# Optimal Response-Linear Achievable Iterpolator

# TODO: upload package to pypi

This repo contains the implementation of the optimal response-linear achievable interpolator from https://arxiv.org/abs/2110.11258 in python.

## Installation

Install the python package through pip by `pip install orlai`.

## Usage

The following is implemented in `example.ipynb`.

### Generate a covariance matrix

```
import numpy as np
n = 300 # number of datapoints
d = 600 # dimension
c = np.eye(d)
for i in range(d):
    for j in range(d):
            c[i,j] = 0.5**(abs(i-j))
```

### Generate features

`
X_train = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = n)
X_test = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = n/10)
`

### Generate data according to `y = X.dot(w_star) + xi`

```
w_star = np.random.multivariate_normal(mean = np.zeros(d),cov = np.eye(d))
y_train = X_train.dot(w_star) + np.random.multivariate_normal(mean = 0,cov = 1, size = n)
y_test = X_test.dot(w_star) + np.random.multivariate_normal(mean = 0,cov = 1, size = n/10)
```

# Fit and predict
```
orlai = Orlai()
orlai.fit(X_train)
y_test_orlai = orlai.predict(X_test)
```

# Fit and predict for minimum-norm-interpolator
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
y_test_reg = reg.predict(X_test)
```

# Compare

```
print(np.linalg.norm(y_test - y_test_orlai)**2)
print(np.linalg.norm(y_test - y_test_reg)**2)
```


