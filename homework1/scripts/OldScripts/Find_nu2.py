"""!
@brief parameter nu2 estimation.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

# imports
import TF2D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy

dataset_p = np.load('signals_3_patients.npy')

nu_est = []

for ind_s in range(3):

    nu_2_guess = 0.012

    def fun_MSE(nu_2_):

        T = 450
        signals_num_simulation = TF2D.solver(nu_2_,T)
        MSE = mean_squared_error(signals_num_simulation[0,0,:],dataset_p[ind_s,0,:T+1])
        print('MSE %f guess %f' %(MSE,nu_2_))
        return MSE

    res = scipy.optimize.minimize(
        fun_MSE,
        nu_2_guess,
        method='Nelder-Mead',
        bounds = scipy.optimize.Bounds(0.0116, 0.0124),
        options = {"maxiter": 10}
    )

    nu_est.append(res.x)

    print("Final estimate: %f \n"  %nu_est[-1])
