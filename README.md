# SCT-Challenges
Scientific Computing Tools for Mathematical Modelling - Challenges

## Homework1

Anti-tachycardia pacing (ATP) delivers pacing pulses to interrupt a tachyarrhythmia episode and restore normal sinus rhythm. Devices such as the implantable cardioverter-defibrillator (ICD) employ this technique before delivering high-voltage shocks to reduce the patient's sensation of pain and extend the life of the device's battery.

To ensure that ATP is effective, these devices must be programmed by selecting the proper pacing pulse width (fixed in the simulations), duration and timing. Note that it is preferred to keep the impulse limited, to reduce battery consumption.

The goal of this homework is to identify an optimal ATP strategy based on a single impulse.
The dataset comprises a single bipolar trace (first row), the time discretization (second row), and an impulse shape for three different patients (third row).


This process can be modeled using the monodomain model coupled with a system of ODE given by the Rogers-McCulloch model.
An implementation of a numerical discretization of the model is provided in the file `TF2D.py`.
The numerical simulation comprises two sinus activation followed by an extra-stimulus which triggers a sustained re-entrant circuits. All the physical coefficients of the model are well defined, outside of $\nu_2$, that can vary between $0.0116$ and $0.0124$.

As solver parameters, it is recommended to use the following configurations:
 - $N = 128$, $M = 64$,  $\Delta t=0.01$
 - $N = 256$, $M = 128$,  $\Delta t=0.005$

For the mathematical formulation, three different time windows can be considered:
 - measurement windows $(0,450)\ ms$ used by the device for selecting the impulse characteristics;
 - ATP windows $(450,525)\ ms$ where the impulse (maximum duration $10\ ms$) is delivered;
 - tracking window $(600,800)\ ms$ to verify the effectiveness of the strategy.


## Homework 2

A stroke is a pathological condition in which the blood flow to the brain is stopped.
The most common cause is the formation of a blood clot (thrombus). Most atrial thrombi form in the left atrial appendage (LAA), with greater or lesser incidence depending on the patient morphology. Many studies have investigated this variability using imaging techniques.

Four principal forms (reported in Figure, taken from [this article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118822)) have been identified:
 1. Cactus,
 2. Chicken Wing,
 3. Windsock,
 4. Cauliflower.

The objectives of this homework are:
 1. to develop a 2D shape model that contains the principal forms of LAA.
 2. to numerically approximate a 2D flow in the LAA, assuming a constant velocity at the entrance of the LAA in the direction perpendicular to the LAA main axis (adaptation of the cavity test case).
 3. define one or more indicators based on approximated velocity distribution that might help predict the risk of ischemic stroke formation.

## Homework 3


The COVID 19 pandemic required continuous monitoring by authorities due to several changes in the scenario that occurred. Predictive models of contagion evolution may be pivotal to making weighted decisions regarding actions to be implemented. In this homework, you will develop a physics-based or a data-based technique, allowing for the prediction of the pandemic trend in the following 7 days starting from daily acquired data.

Objectives of the homework:

1. **Step 1**: set up an automatic tool that accesses, updates and organizes the COVID-19 epidemiological data of Italy (on a regional basis);
3. **Step 2**: implement a strategy to update daily your prediction of the trend for the following days.

The prediction will involve the following categories (on a regional basis):
 1. new daily infections
 2. hospitalized
 3. recovered
 4. deceased.

Data can be extracted from the [database  dataset repository](https://github.com/pcm-dpc/COVID-19).

## Homework 4

The aim of this homework is to explore the potential of Graph Neural Networks and, in particular, try to understand
if and where they could be useful in solving partial differential equations’ problems.

As a matter of fact, the mesh of the problem can be easily considered as a graph, where each vertex of the mesh
is a node in the graph and each connection between two vertices an edge.

As a further note, in a framework like this it is not possible to implement a simple Convolutional Neural
Network: in fact, a mesh does not have a fixed structure like, for example, an image where every pixel has the
same amount of neighboring pixels, distributed over a fixed grid. 

Meshes of PDEs problem, instead, not only
differ in structure, but even nodes of the same mesh will have different connectivities, thus making impossible
to use the traditional convolution operator. This is where Graph Convolution comes into play: it generalizes
the convolution operator to irregular domains.

All this considered, using particular architectures that will be illustrated in this report, we want to see if adding
the connectivity of the mesh as a feature fed into a neural network, can improve the accuracy in the estimation
of the solution.

