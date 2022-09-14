"""!
@brief Final optimization step for the duration fixed the timing.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

# imports
import TF2D_start_450

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from scipy.stats import qmc

from bayes_opt import BayesianOptimization


dataset_p = np.load('signals_3_patients.npy')

def plot_bo(f, bo):
    x = np.linspace(0.0, 5.0, 100000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    print(" Il minimo della funzione stimata è [duration, MSE]:", x[np.argmin(-mean)], np.min(-mean))
    plt.figure(figsize=(16, 9))
    #plt.plot(x, f(x))
    plt.plot(x, -mean)
    plt.fill_between(x, - mean + sigma, - mean - sigma, alpha=0.1)
    #plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.scatter(bo._gp.X_train_.flatten(), - bo.space.target[:len(bo._gp.X_train_.flatten())], c="red", s=50, zorder=10)
    plt.show()




nu_2 =  0.01234  # Value of nu_3 for the desired patient
ind_s = 0 # patient [0, 1, 2]
coeff = 1 # Coefficient for the duration in the loss
timing = 474.05 # Fixed timing of the impulse


signals_file_name = 'patient_' + str(ind_s+1) + '/signals_fixed_nu_patient_' + str(ind_s+1) + '.npy'
signals_init = np.load(signals_file_name)  # NOT NORMALIZED (obtainded thanks to TF2D_generate_initial_condition.py)
Ut = np.load('patient_' + str(ind_s+1) + '/Ut_450.npy')  # generally do not modify
Wt = np.load('patient_' + str(ind_s+1) + '/Wt_450.npy')  # generally do not modify
T_init = 450  # initial time instant of simulation (generally do not modify)


def fun_MSE(duration):

    T = 800
    num_simulation, num_simulation_normalized = TF2D_start_450.solver(ind_s, nu_2, T, signals_init, Ut, Wt, T_init, timing, duration)
    MSE = mean_squared_error(num_simulation_normalized[0,0,700:], np.zeros(num_simulation_normalized[0,0,700:].shape)) + coeff * (duration/ 10.0)**2
    print('MSE %f guess time = %f guess duration = %f' %(MSE,timing,duration))
    return - MSE


# Bounded region of parameter space
upper_bound = 1.4049 # Duration obtained by the joint optimization with the timing
pbounds = {'duration': (0, upper_bound)}

optimizer = BayesianOptimization(
    f = fun_MSE,
    pbounds = pbounds,
    verbose = 2,  # 2 prints all,  1 prints only when a maximum is observed, 0 is silent
    random_state = 1,
)


# Sample from a Sobol sequence 2^m couples [duration, impulse]
sampler = qmc.Sobol(d=1, scramble=True)
sample = sampler.random_base2(m=3)

# Lower and Upper bounds for rescaling (we explore the region for duration < 5)
l_bounds = [0.0]
u_bounds = [upper_bound]
points = qmc.scale(sample, l_bounds, u_bounds)

# Visit the points sampled from the Sobol sequence
for k in range(points.shape[0]):
    optimizer.probe(
        params={"duration": points[k, 0]},
        lazy=True
    )

# Perform Bayesian optimization
optimizer.maximize(
    init_points = 0,
    n_iter = 10
)

print("Final estimate:",  optimizer.max)

plot_bo(fun_MSE, optimizer)
