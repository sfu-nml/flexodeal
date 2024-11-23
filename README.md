# What is Flexodeal?

Flexodeal is a computational library designed to perform three-dimensional dynamic or quasi-static deformations of skeletal muscle tissue using a Hill-type muscle model and the finite element library [deal.II](https://www.dealii.org).

> :warning: For now, muscle geometries can only contain muscle tissue with intramusucular fat, i.e. no tendon or aponeurosis. Support for these quantities will come in a future version of Flexodeal.

# Installation

1. Download and compile deal.II (available for Linux and MacOS). Flexodeal started its development using v9.3.0 of deal.II, but it has been to show to work on the latest version as well (v9.6.0).

2. Download the [latest release](https://github.com/javieralmonacid/flexodeal/releases) of Flexodeal using the .zip file or simply clone this repository to get bleeding edge updates by simply invoking ```git pull```.

3. Go to the directory where you extracted or cloned Flexodeal and compile using CMake: ```cmake . -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=<path/to/deal.II>```. This will set the appropriate dependencies to then ```make``` the code.

# How to use it

In general, Flexodeal should be treated as modelling framework and not as a "works out of the box" solution. While some tensile experiments can be performed with minimal modifications, you should also be prepared to modify the code directly. Learn how to do this by learning more about how deal.II works: [https://www.dealii.org/current/doxygen/deal.II/Tutorial.html](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html).

## 1. Setting up the mesh and boundary IDs

As usual in deal.II applications, Flexodeal requires a structured mesh made of hexahedral elements (hanging nodes are not supported at the moment). We call each hexahedral element in the mesh a *voxel*.

You may generate the mesh in a different software and export it as a .msh file or make it using deal.II functions (see ```Solid<dim>::make_grid```). Either way, make sure that you have assigned the proper boundary IDs, since this will affect the way the boundary conditions are imposed in ```Solid<dim>::make_constraints```.


## 2. Setting up the quadrature point table (requires a mesh)

The main difference with [Flexodeal Lite](https://github.com/sfu-nml/flexodeal-lite) is that here it is mandatory to use a quadrature point (QP) file. By default, in ```parameters.prm```, this file is called ```quadrature_point_data.csv``` (see ```set QP list filename``` in the ```Materials``` subsection).

In the context of finite element methods, many of the integrals involved in computing element stiffness matrices, mass matrices, and load vectors are not straightforward to evaluate analytically. Since exact integration may not be possible or practical for most element shapes, numerical integration is used to approximate these integrals.

Quadrature points are used to evaluate these integrals approximately by placing "evaluation points" within each element and using weights associated with these points to compute the integral. This helps in ensuring that the element-level calculations (like stiffness or mass matrices) are accurately represented in the global system, which is essential for the solution of the finite element problem. 

The number of quadrature points is determined by the *order* of the quadrature rule. Flexodeal uses a [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature) rule and the order is specified in the ```parameters.prm``` file (see ```set Quadrature order``` in the ```Finite element system``` subsection). Below is a guideline to choose the correct order based on the chosen polynomial degree and whether the simulation is quasi-static or dynamic:

| ```set Polynomial degree``` | ```set Type of simulation``` | ```set Quadrature order```| Number of QPs per voxel |
|:---------------------------:|:----------------------------:|:-------------------------:|:-----------------------:|
| 1 | quasi-static | 2 | 8   |
| 1 | dynamic      | 2 | 8   |
| 2 | quasi-static | 3 | 27  |
| 2 | dynamic      | 4 | 64  |
| 3 | quasi-static | 4 | 64  |
| 3 | quasi-static | 5 | 125 |


The QP file serves two purposes: it lists the quadrature points and attaches QP-dependent material properties. To first generate the file, ```make``` the code and then use one of the following options:
* If reading the mesh from a file, say ```mesh.msh```, call
```
./flexodeal -QP_LIST_ONLY -MESH_FILE=mesh.msh
```
* If using the ```Solid<dim>::make_grid``` function already implemented in ```flexodeal.cc``, call
```
./flexodeal -QP_LIST_ONLY
```
This should generate the file ```quadrature_point_data.csv``` (or whatever name you chose in ```parameters.prm```) in the current directory. The file will only contain three columns: ```qp_x```, ```qp_y```, and ```qp_z```, denoting each one of the components of the QP.

The next step is to attach physiological properties to each one of these points. To do so, edit the CSV file and insert a new column with the name of the property and the value for each point. At the moment, the following properties are supported (written as ```column header```: description [units]):
* ```max_iso_stress_muscle```: maximum isometric stress of muscle [Pa].
* ```muscle_fibre_orientation_x```: Initial fibre orientation vector (normalized), x component.
* ```muscle_fibre_orientation_y```: Initial fibre orientation vector (normalized), y component.
* ```muscle_fibre_orientation_z```: Initial fibre orientation vector (normalized), z component.
* ```tissue_id```: Tissue ID (```unsigned int```) to differentiate different parts of the muscle.
* ```fat_fraction```: Fraction of intramuscular fat present in muscle. It is a number between 0 and 1.

> :warning: *The order of these columns does not matter, however, it is important to use the correct header name, otherwise, the code will throw an error.*

## 3. Running the code

Once you have set up your quadrature point file, you may run Flexodeal as
```
./flexodeal
```
This assumes that your activation profile is given in ```control_points_activation.dat```, your boundary strain in ```control_points_strain.dat``` and your parameters in ```parameters.prm```. If you want to use files with other names, you can use flags to override these default settings:
```
./flexodeal -PARAMETERS=other_parameters.prm -ACTIVATION=another_activation.dat -BDY_STRAIN=another_boundary_strain.dat
```
You can also change the output directory by adding the ```-OUTPUT_DIR``` flag. For instance
```
./flexodeal -OUTPUT_DIR=my_favourite_name_for_a_folder
```
You may combine these flags as needed. If you are reading a mesh from file, remember to attach this flag as well:
```
./flexodeal -MESH_FILE=an_awesome_mesh.msh
```