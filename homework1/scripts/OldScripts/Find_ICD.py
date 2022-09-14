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

nu_2 = [0.0124, 0 , 0.011831] # Da mettere i nu giusti
time_est = []
duration_est = []
coeff = 0.5

for ind_s in range(1):

    time_guess = 488
    duration_guess = 5

    def fun_MSE(x):

        T = 800
        signals_num_simulation = TF2D.solver(nu_2[ind_s],T, x[0], x[1])
        MSE = mean_squared_error(signals_num_simulation[0,0,600:], np.zeros(signals_num_simulation.shape)) + coeff * (x[1] / 10.0)**2
        print('MSE %f guess time = %f guess duration = %f' %(MSE,x[0],x[1]))
        return MSE

    res = scipy.optimize.minimize(
        fun_MSE,
        np.array([time_guess, duration_guess]),
        method = 'Nelder-Mead',
        bounds = scipy.optimize.Bounds([450, 0], [525, 10]),
        options = {"maxiter": 10}
    )

    time_est.append(res.x[0])
    duration_est.append(res.x[0])

    print("Final estimate: time = %f ms, duration = %f \n"  %(time_est[-1], duration_est[-1]))
