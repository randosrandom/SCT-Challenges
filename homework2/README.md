# Homework 2

## :book: Report
In `report` you find a short report about the project.

## :spider_web: Numerical Simulations
In `simulations` folder there are three sub-folders:
- in the folder `freefem_code` we save the *FreeFem++* script (`.edp` extension) through which we compute the numerical solutions to perform the risk assessment (main script: `solversupg.edp`);
- in the folders `photos_simulations` and `vtu_simulations` you find some simulations obtained using *ParaView*.

## :gear: Scripts

In `scripts` folder you find some `.py` files:

1. `Mesh.py` is a small custom library based on `MeshPy` used to generate meshes on a given closed shape;
2. `RBF.py` contains the implementation of our shape model, which uses *Radial Basis Functions*;
3. `control_points_generation.py` and `transformations.py` are custom libraries to manage the position of the control points and shape transformations;
4. `main.py` generates the shapes and meshes analysed in this work;
5. `main_funnel_slope_generation.py` is a script that generates new custom shapes, used to prove the crucial role played by the *funnel* and *slope* effects in the risk assessment (see report for details).

In the sub-folder `geometries/new_meshes` we save the meshes (`.msh` extension) created by the scripts above.  

## :books: References

[1] Otso Arponen Miika Korhonen, Antti Muuronen et al. Left atrial appendage morphology in patients with
suspected cardiogenic stroke without known atrial fibrillation. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118822#abstract09.

[2] F. Hecht. New development in freefem++. J. Numer. Math., 20(3-4):251–265, 2012. ISSN 1570-2820. URL
https://freefem.org/.

[3] Andreas Kloeckner. MeshPy: Simplicial Mesh Generation from Python. https://github.com/inducer/meshpy


## :thought_balloon: Authors
- Galvan Lorenzo ([@lorenzogalvan](https://github.com/lorenzogalvan))
- Leimer Caterina ([@caterinaleimer](https://github.com/caterinaleimer))
- Singh Randeep ([@rando](https://github.com/randosrandom))
