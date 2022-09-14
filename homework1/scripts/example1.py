"""!
@file example.py

@brief import data.

@author Stefano Pagani <stefano.pagani@polimi.it>.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt

dataset_p = np.load('signals_3_patients.npy')
dataset_p_sim = np.load('patient_3/signals_normalized_ICD_patient_3.npy')

for ind_s in [2]:
    plt.plot(dataset_p[ind_s,1][:],dataset_p[ind_s,0][:])
    plt.plot(dataset_p_sim[0, 1][:], dataset_p_sim[0, 0][:])
    plt.title("Patient 3: activation time = 507, duration = 1.60")

plt.show()
