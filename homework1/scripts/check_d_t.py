# Import libraries for simulation
import TF2D

import tensorflow as tf
import numpy as np

# Imports for visualization
from tqdm import tqdm
import matplotlib.pyplot as plt

#Modifica per me
from IPython.display import clear_output


# nu_2 of all the patients
nu2 = [0.0123414, 0.01191 , 0.01171]

# select patient number
patient_number = 0

# optimal values of time and duration correspondent to the patient
time = 481.139470
duration = 1.015920

# Define all the variables used in the simulation
T = 800

# Define the nu2 used in the solver as a perturbation of the nu2 found
time_sim = round(time,1) # round the time to the first decimal value
duration_sim = round(duration,1) # roudn the duration to the first decimal value
#time_sim = 1.01 * time
#time_sim = 0.99 * time
#duration_sim = 1.01 * duration
#duration_sim = 0.99 * duration

nu2_sim = nu2[patient_number]

TF2D.solver(nu2_sim, T, time_sim, duration_sim)

dataset_p = np.load('signals_3_patients.npy')
dataset_p_sim = np.load('signals_num_simulation.npy')


plt.plot(dataset_p[patient_number-1,1][:],dataset_p[patient_number-1,0][:])
plt.plot(dataset_p_sim[0, 1][:], dataset_p_sim[0, 0][:])
plt.title("Patient 1: activation time = " + str(time) + ", duration = " + str(duration))

plt.show()
