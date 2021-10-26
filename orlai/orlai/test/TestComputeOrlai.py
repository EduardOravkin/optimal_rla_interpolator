import unittest
import random
import time
from numpy.testing import assert_array_equal
from orlai.utils.generate_example import generate_example
from orlai.utils.compute_orlai import compute_orlai, compute_orlai_multi


class TestComputeOrlai(unittest.TestCase):

    def test_compute_orlai_multi(self):
        # fix random seed for reproducibility
        random.seed(10)
    
        # generate an example of a linear regression problem
        n, d, c, prior_cov, X_train, X_test, y_train, y_test, w_star = generate_example()

        # compare orlai computed in a single processing way and multiprocessing way
        crossval_param = 10
        snr = 2
        s1 = time.time()
        w_single = compute_orlai(X_train,y_train,c,prior_cov,snr,crossval_param)
        e1 = time.time()
        print(f'single took {e1-s1}')
        s2 = time.time()
        w_multi = compute_orlai_multi(X_train,y_train,c,prior_cov,snr,crossval_param)
        e2= time.time()
        print(f'multi took {e2-s2}')

        speedup = (e1-s1)/(e2-s2)
        #self.assertTrue(speedup>1.5, msg=f"multiprocessing speedup was only {speedup}")
        assert_array_equal(w_single,w_multi, err_msg = f"multiprocessing gives output {w_multi} while single processing gives {w_single}")



