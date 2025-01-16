# Eigenvalue Solvers and Random Schrödinger Operator 

This work contains implementations of numerical methods for solving eigenvalue problems and linear systems. The methods include Conjugate Gradient (CG), Inverse Power Method, and Deflation. The code also includes visualization functions for analyzing the behavior of eigenvalues and eigenvectors, both in one-dimensional (1D) and two-dimensional (2D) cases.

## Requirements

- Julia (version 1.11 or higher)
- The following Julia packages:
  - `LinearAlgebra`
  - `Random`
  - `Plots`
  - `LaTeXStrings`
  - `Printf`

To install the required packages, run the following command in Julia's REPL:
```julia
using Pkg
Pkg.add(["LinearAlgebra", "Random", "Plots", "LaTeXStrings", "Printf"])
```

## Code Structure

The main code is divided into the following sections:
  
### 1. **Helper Functions**
- **`rand_schrodinger_1D`**: Generates a random Schrödinger operator.
- **`schrodinger_1D`**: Generates the Schrödinger operator.
- **`potential_1D`**: Generates a random potential.
- **`Hmatvec`**: Computes the matrix-vector product with a random Schrödinger operator in 2D based on the 1D case.

### 2. **Eigenvalue/Linear Systems Solvers**
- **`CG (1D and 2D)`**: Implements the Conjugate Gradient method to solve linear systems.
- **`inverse_power_method (1D)`**: Implements the Inverse Power Method for computing the smallest eigenvalue.
- **`deflation (1D and 2D)`**: Implements the deflation method to find multiple eigenvalues and eigenvectors.

### 3. **Visualization**
- **`plot_convergence`**: Plots the convergence of the Inverse Power Method.
- **`plot_relation_1D`**: Visualizes the solution and eigenvectors for the 1D case.
- **`plot_relation_2D`**: Visualizes the solution and eigenvectors for the 2D case.

## Usage

### Test

To run the main tests, verifying the correctness of each algorithm, execute the following function:
```julia
test()
```

### Visualizations

To visualize the convergence of the Inverse Power Method:
```julia
plot_convergence()
```

To visualize the solution and eigenvectors for the 1D case:
```julia
plot_relation_1D()
```

To visualize the solution and eigenvectors for the 2D case with different grid sizes:
```julia
plot_relation_2D()
```


