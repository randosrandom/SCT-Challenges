# Homework 4 - Graph Neural Networks (GNN)

## :memo: Description

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

## :book: Report
In `report` folder you find a short report on the project.

## :spider_web: Numerical Simulation
In `mesh` folder there are FreeFem++ script (`.edp` extension) where the numerical solution is computed through finite elements. Then u functions u and f are saved in two `.txt` file and the mesh is saved in `.msh` format.

In the subfolders are stored the mesh's data rewritten in a more useful form for our scripts through `graph.py` (see below).

Each mesh contains 4 different `.npy` files containing coordinates of the points, edges and triangles numeration and the adjacency matrix computed with the previous information.

## :gear: Neural Networks

In `src` folder there are 2 Python Notebooks and a Python file
1. `graph.py` is a small custom library based on `networkx` for generate and handle graphs
2. `GAT_fully_connected.ipynb` is the implementation of our GNN with static Graph Attention (GAT) layers and a FCNN
3. `GATv2_fully_connected.ipynb` is the implementation of our GNN with dynamic Graph Attention (GATv2) layers and a FCNN

There are also other structures we tried with small to no success and a folder with dummy netorks for comparison of results. 

## :books: References

[1] Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning mesh-based
simulation with graph networks. 2020. doi: 10.48550/ARXIV.2010.03409. URL https://arxiv.org/abs/
2010.03409.

[2] Wenzhuo Liu, Mouadh Yagoubi, and Marc Schoenauer. Multi-resolution graph neural networks for pde
approximation. In Igor Farkaš, Paolo Masulli, Sebastian Otte, and Stefan Wermter, editors, Artificial
Neural Networks and Machine Learning – ICANN 2021, pages 151–163, Cham, 2021. Springer International
Publishing. ISBN 978-3-030-86365-4.

[3] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio.
Graph attention networks, 2017. URL https://arxiv.org/abs/1710.10903.

[4] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks?, 2021. URL
https://arxiv.org/abs/2105.14491.

[5] F. Hecht. New development in freefem++. J. Numer. Math., 20(3-4):251–265, 2012. ISSN 1570-2820. URL
https://freefem.org/.

[6] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR
Workshop on Representation Learning on Graphs and Manifolds, 2019.
10

## :thought_balloon: Authors 
- Andrea Novellini ([@geonove](https://github.com/geonove))
- Randeep Singh ([@rando98](https://github.com/randosrandom))
- Luca Sosta ([@sostaluca](https://github.com/SostaLuca98))