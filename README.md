# Helmholtz Solver
A solver for the Helmholtz equation.

## Abstract
The Generalized Minimization of Residuals (GMRES) method is a powerful method
for solving linear systems. Nevertheless it may have some restrictions and disabilities
that arise from the complex nature of the systems we are dealing with. The 2-D
Helmholtz equation with a spatially dependent wave number is one of these systems.
Solving such a system may be very slow (when feasible). In this project, we seek
to find the best methods for choosing preconditioners for solving the heterogeneous
Helmholtz equation using GMRES (with varying preconditioner). We test a wide
range of options for realistic grids for the wave number, for a range of grid sizes,
using a robust and generic algorithm.

## Algorithm

We use Green's function in order to find the solution for in-homogeneous Helmholtz equation.
The Green's function (denoted as G), is the solution for the following equation:

![WhatsApp Image 2022-08-28 at 12 04 31](https://user-images.githubusercontent.com/73799544/187066379-c0cf65a0-04ed-4221-8399-4ed27fb86187.jpeg)

For the case of Helmholtz equation (L), Green's function looks as follow:

![WhatsApp Image 2022-08-28 at 11 59 15](https://user-images.githubusercontent.com/73799544/187066308-a53700d8-c95c-4a5f-a97a-4845b962e54f.jpeg)

Notice that it's the solution for the following equation:


