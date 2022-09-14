"""!
@brief parameter nu2 estimation.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

# imports
import TF2D_start_450

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

from bayes_opt import BayesianOptimization


dataset_p = np.load('signals_3_patients.npy')

def cartesian_product(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def create_grid(bo):
    x1 = np.linspace(450, 525, 1000)
    x2 = np.linspace(0, 10, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    mean, sigma = bo._gp.predict( cartesian_product(x2, x1), return_std=True)
    return X1, X2, -mean


def plot_bo_3D(bo):

    X1, X2, mean = create_grid(bo)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X1, X2, mean.reshape(1000, 1000), cmap= 'viridis')
    plt.xlabel('Activation time')
    plt.ylabel('Duration')
    ax.set_zlabel('MSE + penalty')
    plt.show()


def plot_bo_2D(bo):

    X1, X2, mean = create_grid(bo)

    fig = plt.figure()
    plt.contourf(X1, X2, mean.reshape(1000, 1000))
    plt.colorbar()
    plt.xlabel('Activation time')
    plt.ylabel('Duration')
    plt.title('MSE + penalty')
    plt.show()



nu_2 = [0.01234, 0.01191 , 0.01171] # Da mettere i nu giusti
coeff = 0.5

for ind_s in [2]:

    signals_file_name = 'patient_' + str(ind_s+1) + '/signals_fixed_nu_patient_' + str(ind_s+1) + '.npy'
    signals_init = np.load(signals_file_name)  # NOT NORMALIZED (obtainded thanks to TF2D_generate_initial_condition.py)
    Ut = np.load('patient_' + str(ind_s+1) + '/Ut_450.npy')  # generally do not modify
    Wt = np.load('patient_' + str(ind_s+1) + '/Wt_450.npy')  # generally do not modify
    T_init = 450  # initial time istant of simulation (generally do not modify)


    def fun_MSE(impulse, duration):

        T = 800
        num_simulation, num_simulation_normalized = TF2D_start_450.solver(ind_s, nu_2[ind_s], T, signals_init, Ut, Wt, T_init, impulse, duration)
        MSE = mean_squared_error(num_simulation_normalized[0,0,600:], np.zeros(num_simulation_normalized[0,0,600:].shape)) + coeff * (duration/ 10.0)**2
        print('MSE %f guess time = %f guess duration = %f' %(MSE,impulse,duration))
        return - MSE


    # Bounded region of parameter space
    pbounds = {'impulse': (450, 525), 'duration': (0, 10)}

    optimizer = BayesianOptimization(
        f = fun_MSE,
        pbounds = pbounds,
        verbose = 2,  # 2 prints all,  1 prints only when a maximum is observed, 0 is silent
        random_state = 1,
    )

    optimizer.maximize(
        init_points = 10,
        n_iter = 10
    )


    print("Final estimate:",  optimizer.max)

    plot_bo_3D(optimizer)
    plot_bo_2D(optimizer)

