# What is Flexodeal?

Flexodeal is a computational library designed to perform three-dimensional dynamic or quasi-static deformations of skeletal muscle tissue using a Hill-type muscle model and the finite element library [deal.II](https://www.dealii.org).

> :warning: For now, muscle geometries can only contain muscle tissue with intramusucular fat, i.e. no tendon or aponeurosis. Support for these quantities will come in a future version of Flexodeal.

The underlying material properties, numerical algorithms, are described in the following paper:

* Almonacid, J. A., Domínguez-Rivera, S. A., Konno, R. N., Nigam, N., Ross, S. A., Tam, C., & Wakeling, J. M. (2024). A three-dimensional model of skeletal muscle tissues. SIAM Journal on Applied Mathematics, S538-S566. [https://doi.org/10.1137/22M1506985](https://doi.org/10.1137/22M1506985)

This coding framework has been used in the following studies:

* Ross, S. A., Domínguez, S., Nigam, N., & Wakeling, J. M. (2021). The Energy of Muscle Contraction. III. Kinetic Energy During Cyclic Contractions. Frontiers in Physiology, 12(April), 1–16. https://doi.org/10.3389/fphys.2021.628819

* Konno, R. N., Nigam, N., & Wakeling, J. M. (2021). Modelling extracellular matrix and cellular contributions to whole muscle mechanics. PLoS ONE, 16(4 April 2021), 1–20. https://doi.org/10.1371/journal.pone.0249601

* Ryan, D. S., Domínguez, S., Ross, S. A., Nigam, N., & Wakeling, J. M. (2020). The Energy of Muscle Contraction. II. Transverse Compression and Work. Frontiers in Physiology, 11(November), 1–15. https://doi.org/10.3389/fphys.2020.538522

* Wakeling, J. M., Ross, S. A., Ryan, D. S., Bolsterlee, B., Konno, R., Domínguez, S., & Nigam, N. (2020). The Energy of Muscle Contraction. I. Tissue Force and Deformation During Fixed-End Contractions. Frontiers in Physiology, 11, 1–42. https://doi.org/10.3389/fphys.2020.00813

* Domı́nguez S. From eigenbeauty to large-deformation horror. Ph.D. Thesis, Simon Fraser University. 2020. Available from: http://summit.sfu.ca/item/20968

* Ross, S. A., Ryan, D. S., Dominguez, S., Nigam, N., & Wakeling, J. M. (2018). Size, history-dependent, activation and three-dimensional effects on the work and power produced during cyclic muscle contractions. Integrative and Comparative Biology, 58(2), 232–250. https://doi.org/10.1093/icb/icy021

* Rahemi, H., Nigam, N., & Wakeling, J. M. (2015). The effect of intramuscular fat on skeletal muscle mechanics: implications for the elderly and obese. Journal of The Royal Society Interface, 12(109), 20150365. [https://doi.org/10.1098/rsif.2015.0365](https://doi.org/10.1098/rsif.2015.0365)

* Rahemi, H., Nigam, N., & Wakeling, J. M. (2014). Regionalizing muscle activity causes changes to the magnitude and direction of the force from whole muscles—a modeling study. Frontiers in physiology, 5, 298. [https://doi.org/10.3389/fphys.2014.00298](https://doi.org/10.3389/fphys.2014.00298)

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

Review the file ```quadrature_point_data.csv``` to see how the QP file should look like.

### New in version 0.2.1!

To avoid opening and saving the file in an external program such as Microsoft Excel or LibreOffice Calc, you can use the bash file ```add_columns_to_qp_file.sh``` to add the necessary columns to the CSV file. For instance, the following command:
```
bash add_columns_to_qp_file.sh quadrature_point_data.csv max_iso_stress_muscle 200000 muscle_fibre_orientation_x 0.939692620785908 muscle_fibre_orientation_y 0 muscle_fibre_orientation_z 0.342020143325669 fat_fraction 0
```
would add the columns ```max_iso_stress```, ```muscle_fibre_orientation_x```, ```muscle_fibre_orientation_y```, ```muscle_fibre_orientation_z```, and ```fat_fraction``` to the file ```quadrature_point_data.csv``` with values 200000, 0.939692620785908, 0, 0.342020143325669, and 0, respectively.

## 3. Set up markers

You may set a list of markers to track displacements at different points in the geometry. The list (by default, `markers.dat`, with the filename set in `parameters.prm` in the `Measuring locations` subsection) has four columns: the first one is a label and the other are the three components of the marker. Note that, in this context, a marker is a mesh vertex that contains displacement degrees of freedom. Therefore, every marker **must** be a vertex in the mesh. Check the file `markers.dat` to see the structure of this file. **If you do not know the location of any markers, just create an empty file with the name as given in `set Markers list file` inside `parameters.prm`.**

## 4. Running the code

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

## Output using binary files: what they are and how to read them

A binary file is a file that contains data in a format that is not directly readable by humans. Unlike text files, which store data as plain text (ASCII or Unicode), binary files store data in raw binary format, which is optimized for computer processing rather than human readability.

Each of these files (which are written at each time step) contains a two-dimensional array (number of quadrature points) x (number of columns),
where each column represents a different quantity of interest and each row is a different quadrature point.

Since these are rather large files, these are **not** output by default. To export these files, set in `parameters.prm`:
```
subsection Output binary files
    set Output binary files main variables = true
    set Output binary files tensors = true
end
```

The first type of files exports binary files named `cell_data_main-3d-XYZ.data`, where XYZ is the time step. Each one of these files contains the following columns (20 in total):
1. `qp_x` (X component of the quadrature point in the reference configuration)
2. `qp_y` (Y component of the quadrature point in the reference configuration)
3. `qp_z` (Z component of the quadrature point in the reference configuration)
4. `JxW` (dilation J x quadrature point weight)
5. `det_F` (determinant of the deformation tensor)
6. `u1` (First component of current displacement)
7. `u2` (Second component of current displacement)
8. `u3` (Third component of current displacement)
9. `v1` (First component of current velocity)
10. `v2` (Second component of current velocity)
11. `v3` (Third component of current velocity)
12. `p` (Current pressure)
13. `D` (Current dilation)
14. `stretch` (Fibre stretch)
15. `stretch_bar` (Fibre stretch, isochoric component)
16. `strain_rate` (Fibre strain rate, normalized)
17. `strain_rate_bar` (Fibre strain rate, normalized, isochoric component)
18. `orientation_x` (X component of current fibre orientation)
19. `orientation_y` (Y component of current fibre orientation)
20. `orientation_z` (Z component of current fibre orientation)

The second type of files are named `cell_data_tensors-3d-XYZ.data` and contains the following columns (see the comments in the parameter file to view the exact name of each column, 68 columns in total):
- Column 1: `qp_x`
- Column 2: `qp_y`
- Column 3: `qp_z`
- Column 4: `Jxw`
- Column 5: `det_F`
- Columns 6-14: `F_i_j`, the (i,j) component of the deformation tensor `F`
- Columns 15-23: `tau_i_j`, the (i,j) component of the Kirchhoff stress `tau`
- Columns 24-32: `tau_vol_i_j`, the (i,j) component of `tau_vol`
- Columns 33-41: `tau_iso_i_j`, the (i,j) component of `tau_iso`
- Columns 42-50: `tau_muscle_active_i_j`, the (i,j) component of `tau_muscle_active`
- Columns 51-59: `tau_muscle_passive_i_j`, the (i,j) component of `tau_muscle_active`
- Columns 60-68: `tau_muscle_base_i_j`, the (i,j) component of `tau_muscle_base`

To read these files in Matlab, use the following function:
```
function df = read_binary(filename, ncols)
% READ_BINARY Transforms a the contents of a binary file into a matrix
%
% df = read_binary(filename, cols) reads the file "filename" assuming that
% it will find a vector of "float32" numbers. Assuming this vector has N
% elements, the array is then reshaped to a N/ncols x ncols matrix, where
% ncols is the number of columns the array is expected to have.

fid = fopen(filename);
df = fread(fid,Inf,"float32");
df = reshape(df', ncols, length(df)/ncols)';
fclose(fid);
```