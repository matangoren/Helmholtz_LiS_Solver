# Helmholtz_Solver
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


