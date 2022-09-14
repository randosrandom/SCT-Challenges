# Homework 1

**Authors:** Moreschi Andrea, Singh Randeep, Zanotti Daniela.

# Report folder
* **homework1.pdf**: Short report containing the conceptualization and the mathematical formulation of the problem together with the explanation of the algorithm employed to solve the problem.

# Scripts folder
It collects are the .py scripts necessary to solve the problem. The path is the following:
1. Given a noisy observation of the tachycardic ECG in **signals_3_patients.npy** run the **Find_nu2_bayopt_sobol.py** to estimate the correct value of the parameter $\nu_2$.
2. In each of the three patient's folder (e.g. **patient_1\\**) run the script **TF2D_generate_initial_condition.py** in order to avoid at each iteration of the optimization procedure to perform a simulation of the first $450$ ms. The parameters that are needed to the solver are the value of $\nu_2$ and the number of the patient.
3. To obtain an estimate of the correct timing and the optimal duration of the impulse, run **Find_ICD_bayopt_sobol.py**. The parameter that need to be passed is the value of $\nu_2$ obtained at the first step and used to generate the initial conditions.
4. To find the minimal value of the duration in order to stop the tachycardia, fixed the timing found in step 3, run the script **Duration_optimization.py**. The parameters that needed to be modified are the patient index and his value of $\nu_2$ as long as the timing of the impulse and the best duration found in the previous optimization procedure.
5. To check whether the solution found for $\nu_2$ is consistent or not, check if the signal goes to 0 with nu2 $\pm$ 0.01, by running the script **Check_nu2.py**. The parameters that need to be changed are the number of the patient considered and the values of duration and time activation found in the previous optimization procedure.
6. To check whether the solutions found for the duration $d$ and the timing $t$ are consistent, we check if the signal goes to zero with small variations and/or rounded values of the solutions. Run the script **Check_d_t.py**. The parameters that need to be changed are the number of the patient considered along with the correspondent optimal value $\nu_2$ found in the step 1.
