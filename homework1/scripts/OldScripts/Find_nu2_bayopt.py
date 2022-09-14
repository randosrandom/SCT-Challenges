"""!
@brief parameter nu2 estimation.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

# imports
import TF2D

import json
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events


dataset_p = np.load('signals_3_patients.npy')


def plot_bo(f, bo):
    x = np.linspace(0.0112, 0.0128, 100000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    print(" Il minimo della funzione stimata è [nu_2, MSE]:", x[np.argmin(-mean)], np.min(-mean))
    plt.figure(figsize=(16, 9))
    #plt.plot(x, f(x))
    plt.plot(x, -mean)
    plt.fill_between(x, - mean + sigma, - mean - sigma, alpha=0.1)
    #plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.scatter(bo._gp.X_train_.flatten(), - bo.space.target[:len(bo._gp.X_train_.flatten())], c="red", s=50, zorder=10)
    plt.show()


for ind_s in [0]:

    def fun_MSE(nu_2_):

        T = 450
        signals_num_simulation = TF2D.solver(nu_2_,T)
        MSE = mean_squared_error(signals_num_simulation[0,0,:],dataset_p[ind_s,0,:T+1])
        print('MSE %f guess %f' %(MSE,nu_2_))
        return - MSE


    # Bounded region of parameter space
    pbounds = {'nu_2_': (0.0116, 0.0124)}

    optimizer = BayesianOptimization(
        f = fun_MSE,
        pbounds = pbounds,
        verbose = 2,  # 2 prints all,  1 prints only when a maximum is observed, 0 is silent
        random_state = 1
    )

    optimizer.maximize(
        init_points=5,
        n_iter=15
    )


    print("Final estimate:",  optimizer.max)

    plot_bo(fun_MSE, optimizer)
