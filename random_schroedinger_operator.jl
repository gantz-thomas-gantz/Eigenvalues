using LinearAlgebra, Random
using Plots, LaTeXStrings, Printf

# #####################
# One-dimensional case
# #####################

### Exercise 1 ###
"""
Generates a random 1D Schrödinger operator for a system of size `N`.

The operator is constructed as the sum of:
- A symmetric tridiagonal Laplacian matrix `L`.
- A random diagonal potential matrix `v`, with diagonal entries randomly 
  chosen between `1.0` and `1/N^2`.

# Arguments
- `N::Int`: Size of the 1D grid.

# Returns
- `SymTridiagonal`: A symmetric tridiagonal matrix.
"""
function rand_schrodinger_1D(N::Int)::SymTridiagonal
    # Construct the Laplacian matrix `L`
    dv = 2 * ones(N)                 # Main diagonal initialized to 2
    ev = -1 * ones(N-1)              # Off-diagonals initialized to -1
    L = SymTridiagonal(dv, ev)

    # Construct the diagonal potential matrix `v`
    for i in 1:N
        dv[i] = rand((1.0, 1 / N^2)) # Random potential values
    end
    v = Diagonal(dv)

    return L + v
end

"""
Constructs the Laplacian operator `L` for a 1D system of size `N`.

The Laplacian is represented as a symmetric tridiagonal matrix with:
- Main diagonal values set to `2`.
- Off-diagonal values set to `-1`.

# Arguments
- `N::Int`: Size of the 1D grid.

# Returns
- `SymTridiagonal`: A symmetric tridiagonal matrix representing the Laplacian.
"""
function schrodinger_1D(N::Int)::SymTridiagonal
    dv = 2 * ones(N)                 # Main diagonal initialized to 2
    ev = -1 * ones(N-1)              # Off-diagonals initialized to -1
    L = SymTridiagonal(dv, ev)

    return L
end

"""
Generates a random diagonal potential matrix for a 1D system of size `N`.

The diagonal elements are random values chosen between `1.0` and `1/N^2`.

# Arguments
- `N::Int`: Size of the 1D grid.

# Returns
- `Diagonal`: A diagonal matrix representing the potential.
"""
function potential_1D(N::Int)::Diagonal
    dv = zeros(N)
    for i in 1:N
        dv[i] = rand((1.0, 1 / N^2)) # Random potential values
    end
    v = Diagonal(dv)

    return v
end

### Exercise 3 ###
"""
Solves the linear system `Ax = b` using the Conjugate Gradient (CG) method.

# Arguments
- `A::AbstractMatrix`: Coefficient matrix of the system.
- `b::AbstractVector`: Right-hand side vector.
- `eps::Float64`: Tolerance for convergence. Iteration stops when 
  `norm(rk) <= eps`.

# Returns
- `AbstractVector`: Approximate solution vector `x` for `Ax = b`.

# Notes
- If the matrix `A` is not symmetric positive-definite, the results may not 
  converge.
"""
function CG(A::AbstractMatrix, b::AbstractVector, eps::Float64)::AbstractVector
    # Initialize the solution vector with zeros
    x = zeros(length(b))

    # Initial residual `rk` and search direction `p`
    p = b - A * x
    rk_1 = p
    rk = p

    # Iterate until the norm of the residual is below the tolerance `eps`
    while norm(rk) > eps
        # Compute the step size `alpha`
        alpha = dot(rk_1, rk_1) / dot(p, A * p)

        # Update the solution vector `x`
        x += alpha * p

        # Update the residual `rk`
        rk = rk_1 - alpha * A * p

        # Compute the weighting factor `omega`
        omega = dot(rk, rk) / dot(rk_1, rk_1)

        # Update residual and search direction
        rk_1 = rk
        p = rk + omega * p
    end

    return x
end

"""
Computes the smallest eigenvalue and corresponding eigenvector of a symmetric 
matrix `A` using the inverse power method.

The method iteratively refines an eigenvector estimate by solving a series of 
linear systems, leveraging the Conjugate Gradient (CG) method to avoid direct 
matrix inversion.

# Arguments
- `A::AbstractMatrix`: Symmetric matrix for which the smallest eigenvalue is 
  sought.
- `x::AbstractVector`: Initial guess for the eigenvector.
- `eps::Float64`: Tolerance for convergence. Iteration stops when the 
  difference between successive eigenvector estimates is below `eps`.

# Returns
- `Tuple{Float64, Vector{Float64}}`: A tuple containing:
  - Smallest eigenvalue (approximation).
  - Corresponding eigenvector (approximation).

# Notes
- The matrix `A` should be symmetric and invertible for accurate results.
- The function uses the CG method to solve linear systems without direct 
  matrix inversion.
"""
function inverse_power_method(A::AbstractMatrix, x::AbstractVector, eps::Float64)::Tuple{Float64, Vector{Float64}}
    y = x / norm(x)
    x = CG(A, y, 1e-8)
    l = dot(y, x)

    # Iterate until the eigenvector estimate converges
    while norm(y - l * A * y) > eps
        y = x / norm(x)
        x = CG(A, y, 1e-8)
        l = dot(y, x)
    end

    return 1 / l, y
end

### Exercise 4 ###
"""
Computes the `neig` smallest eigenvalues and corresponding eigenvectors of a 
symmetric matrix `A` using the deflation method.

The deflation technique iteratively computes eigenpairs, updating a matrix `P` 
to exclude previously computed eigenvectors from subsequent 
calculations. The Conjugate Gradient (CG) method is used to solve systems 
without explicit matrix inversion.

# Arguments
- `A::AbstractMatrix`: Symmetric matrix for which eigenvalues and eigenvectors 
  are computed.
- `x::AbstractVector`: Initial guess for the eigenvector.
- `eps::Float64`: Convergence tolerance for the relative residual.
- `neig::Int`: Number of smallest eigenvalues to compute.

# Returns
- `Tuple{Vector{Float64}, Matrix{Float64}}`: A tuple containing:
  - `Vector{Float64}`: The `neig` smallest eigenvalues.
  - `Matrix{Float64}`: A matrix where each row corresponds to an eigenvector.

# Notes
- The matrix `A` should be symmetric and invertible.
"""
function deflation(A::AbstractMatrix, x::AbstractVector, eps::Float64, neig::Int)::Tuple{Vector{Float64}, Matrix{Float64}}
    # Initialize dimensions, eigenvalue vector, eigenvector matrix, and deflation matrix
    n_rows, n_cols = size(A)
    Λ = zeros(neig)             # Eigenvalue vector
    Y = zeros(neig, n_cols)     # Eigenvector matrix (each row is an eigenvector)
    P = zeros(n_rows, n_cols)   # Deflation matrix, initialized to zero

    # Loop to compute `neig` eigenpairs
    for n in 1:neig
        y = x / norm(x)
        v = CG(A, y - A * P * y, 1e-8)
        lambda = dot(y, v)

        # Iterate until convergence (relative residual)
        while norm(v - lambda * y) / norm(v) > eps
            y = v / norm(v)
            v = CG(A, y - A * P * y, 1e-8)
            lambda = dot(y, v)
        end

        # Store the computed eigenvalue and eigenvector
        Λ[n], Y[n, :] = 1 / lambda, y

        # Update the deflation matrix to exclude the current eigenvector
        P += lambda * (y * y')  
    end

    return Λ, Y
end

### Exercise 5 ###
"""
Tracks the iterates of the inverse power method, returning all eigenvalue 
approximations and eigenvector estimates for a symmetric matrix `A`.

This function extends the standard inverse power method by keeping a record of 
all eigenvalue and eigenvector estimates at each iteration. The Conjugate 
Gradient (CG) method is used to solve systems without direct matrix inversion.

# Arguments
- `A::AbstractMatrix`: Symmetric matrix for which the smallest eigenvalue and 
  corresponding eigenvector are sought.
- `x::AbstractVector`: Initial guess for the eigenvector.
- `eps::Float64`: Convergence tolerance. Iteration stops when the relative 
  change in the eigenvector is below `eps`.

# Returns
- `Tuple{Vector{Float64}, Vector{Vector{Float64}}}`:
  - `Vector{Float64}`: A vector of all approximations to the smallest eigenvalue.
  - `Vector{Vector{Float64}}`: A vector of eigenvector approximations at each 
    iteration.

# Notes
- The matrix `A` should be symmetric and invertible.
- This method is useful for analyzing convergence behavior.
"""
function inverse_power_method_iterates(A::AbstractMatrix, x::AbstractVector, eps::Float64)::Tuple{Vector{Float64}, Vector{Vector{Float64}}}
    # Initialize storage for eigenvalue approximations (Λ) and eigenvectors (Y)
    Λ = Float64[]                # Array to store eigenvalue estimates
    Y = Vector{Float64}[]        # Array to store eigenvector estimates

    y = x / norm(x)
    x = CG(A, y, 1e-8)
    l = dot(y, x)

    # Store the initial eigenvalue and eigenvector estimates
    append!(Λ, 1 / l)
    push!(Y, y)

    # Iterate until the relative change in the eigenvector is below the tolerance
    while norm(y - l * A * y) > eps
        y = x / norm(x)
        x = CG(A, y, 1e-8)
        l = dot(y, x)

        # Store the current eigenvalue and eigenvector estimates
        append!(Λ, 1 / l)
        push!(Y, y)
    end

    return Λ, Y
end

# #####################
# Two-dimensional case
# #####################

### Exercise 7 ###
"""
Computes the matrix-vector product `y = H * x` for the 2D Schrödinger operator.

This function computes the action of `H` on a vector 
`x` without explicitly constructing `H`. The matrix `H` is defined as:
`H = L ⊗ I + I ⊗ L + diag(v) ⊗ diag(w)`.

# Arguments
- `L::AbstractMatrix`: Symmetric tridiagonal matrix `L` for the 1D Laplacian.
- `v::AbstractMatrix`: Diagonal potential matrix for the first dimension.
- `w::AbstractMatrix`: Diagonal potential matrix for the second dimension.
- `x::AbstractMatrix`: Input vector (stored as a 2D matrix) for the product.

# Returns
- `AbstractMatrix`: Result of the matrix-vector product `H * x`.

# Notes
- The operation is performed without explicitly forming the Kronecker products.
"""
function Hmatvec(L::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix, x::AbstractMatrix)::AbstractMatrix
    N = size(L, 1)  # Assume square matrix `L` of size N x N

    # Compute the contribution from `L ⊗ I + I ⊗ L`
    y = L * x + x * L

    # Add the potential term `diag(v) ⊗ diag(w)`
    for j1 in 1:N
        for j2 in 1:N
            y[j1, j2] += v[j1, j1] * w[j2, j2] * x[j1, j2]
        end
    end

    return y
end

"""
Solves the linear system `H * x = b` using the Conjugate Gradient (CG) method for 2D operators.

This function applies the CG method to solve a system involving a 2D operator `H` 
without explicitly constructing `H`. Instead, the action of `H` 
on a vector is computed using `Hmatvec`.

# Arguments
- `L::AbstractMatrix`: Symmetric tridiagonal matrix `L` for the 1D Laplacian.
- `v::AbstractMatrix`: Diagonal potential matrix for the first dimension.
- `w::AbstractMatrix`: Diagonal potential matrix for the second dimension.
- `b::AbstractMatrix`: Right-hand side matrix representing the 2D input vector.
- `eps::Float64`: Tolerance for convergence. Iteration stops when `norm(rk) <= eps`.

# Returns
- `AbstractMatrix`: Approximate solution matrix `x` for `H * x = b`.

# Notes
- The matrix `H` is not explicitly constructed. Instead, its action is computed 
  on-the-fly using `Hmatvec`.
"""
function CG_2(L::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix, b::AbstractMatrix, eps::Float64)::AbstractMatrix
    # Initialize the solution matrix with zeros
    x = zeros(size(b))

    # Compute the initial residual and search direction
    p = b - Hmatvec(L, v, w, x)
    rk_1 = p
    rk = p

    # Iterate until the residual norm is below the tolerance
    while norm(rk) > eps
        Ap = Hmatvec(L, v, w, p)
        alpha = dot(rk_1, rk_1) / dot(p, Ap)
        x += alpha * p
        rk = rk_1 - alpha * Ap
        omega = dot(rk, rk) / dot(rk_1, rk_1)
        rk_1 = rk
        p = rk + omega * p
    end

    return x
end

### Exercise 9 ###
"""
Computes the smallest `neig` eigenvalues and corresponding eigenvectors of a 
2D Schrödinger operator using deflation and the Conjugate Gradient (CG) method.

This function applies the deflation method to iteratively compute eigenpairs 
of the 2D Schrödinger operator `H`. 

# Arguments
- `L::AbstractMatrix`: Symmetric tridiagonal matrix `L` for the 1D Laplacian.
- `v1::AbstractMatrix`: Diagonal potential matrix for the first dimension.
- `v2::AbstractMatrix`: Diagonal potential matrix for the second dimension.
- `x::AbstractMatrix`: Initial guess for the eigenvector (2D matrix).
- `eps::Float64`: Convergence tolerance for the eigenvector iteration.
- `neig::Int`: Number of smallest eigenvalues and eigenvectors to compute.

# Returns
- `Tuple{Vector{Float64}, Array{Float64, 3}}`:
  - `Vector{Float64}`: Eigenvalues in ascending order.
  - `Array{Float64, 3}`: Eigenvectors stored in a 3D array of size `N x N x neig`.

# Notes
- The matrix `H` is not explicitly constructed. Instead, its action is computed 
  on-the-fly using `Hmatvec`.
"""
function deflation_2(
    L::AbstractMatrix,
    v1::AbstractMatrix,
    v2::AbstractMatrix,
    x::AbstractMatrix,
    eps::Float64,
    neig::Int
)::Tuple{Vector{Float64}, Array{Float64, 3}}
    N = size(L, 1)                # Dimension of the 1D Laplacian
    Λ = zeros(neig)               # Vector to store eigenvalues
    Y = zeros(size(x)..., neig)   # 3D array to store eigenvectors (N x N x neig)
    P = zeros(N * N, N * N)       # Deflation matrix 

    for n in 1:neig
        y = x / norm(x)

        # Apply deflation to the initial guess
        Py = reshape(P * vec(y'), N, N)'  
        v = CG_2(L, v1, v2, y - Hmatvec(L, v1, v2, Py), 1e-6)

        # Compute initial eigenvalue estimate 
        lambda = dot(vec(y'), vec(v'))

        # Iterative refinement of the eigenpair
        while norm(v - lambda * y) / norm(v) > eps
            y = v / norm(v)  
            Py = reshape(P * vec(y'), N, N)'  
            v = CG_2(L, v1, v2, y - Hmatvec(L, v1, v2, Py), 1e-6)
            lambda = dot(vec(y'), vec(v'))  
        end

        # Store the computed eigenvalue and eigenvector
        Λ[n], Y[:, :, n] = 1 / lambda, y

        # Update the deflation matrix `P`
        P += lambda * (vec(y') * vec(y')')  
    end

    return Λ, Y
end

# #####################
# MAIN
# #####################
"""
Main test function to validate the implementation of various algorithms.

This function tests:
1. The Conjugate Gradient (CG) method in 1D (`Exercise 2`).
2. The Inverse Power Method in 1D (`Exercise 3`).
3. The Deflation Algorithm in 1D (`Exercise 4`).
4. The `Hmatvec` function for 2D operators (`Exercise 7`).
5. The Conjugate Gradient method in 2D (`Exercise 8`).
6. The Deflation Algorithm for 2D operators (`Exercise 9`).

Each test computes results using both the custom implementation and 
Julia's built-in methods (where applicable) for comparison.
"""
function test()
    N = 20  # Matrix size for 1D and 2D problems

    # Generate a random Schrödinger operator in 1D
    M = rand_schrodinger_1D(N)
    eigenvalues = eigvals(M)  # Reference eigenvalues using Julia's built-in function

    ### Exercise 2: Test CG ###
    b = ones(N)  # Right-hand side vector
    jl_x = M \ b  # Solve using Julia's backslash operator
    my_x = CG(M, b, 1e-6)  # Solve using custom CG implementation
    println("TEST CG: ", norm(jl_x - my_x) <= 1e-6)

    ### Exercise 3: Test Inverse Power Method ###
    jl_min_eigenvalue = eigenvalues[argmin(abs.(eigenvalues))]  # Closest eigenvalue to zero
    my_l_min, my_y_min = inverse_power_method(M, ones(N), 1e-6)  # Compute using custom method
    println("TEST Inverse Power Method: ", abs(jl_min_eigenvalue - my_l_min) <= 1e-6)

    ### Exercise 4: Test Deflation Algorithm ###
    jl_eigenvalues = sort(eigenvalues, by=abs)[1:5]  # Smallest 5 eigenvalues by magnitude
    my_eigenvalues, my_eigenvectors = deflation(M, ones(N), 1e-6, 5)
    println("TEST Deflation Algorithm: ", norm(jl_eigenvalues - my_eigenvalues) <= 1e-6)

    ### Exercise 7: Test Hmatvec ###
    x = rand(N * N)  # Random input vector for Hmatvec
    x_matrix = reshape(x, N, N)'  # Reshape to 2D matrix
    L = schrodinger_1D(N)  # Generate 1D Laplacian
    v1 = potential_1D(N)  # Generate 1D potential for the first dimension
    v2 = potential_1D(N)  # Generate 1D potential for the second dimension
    @time y_hmatvec = Hmatvec(L, v1, v2, x_matrix)  # Compute using Hmatvec
    H = kron(I(N), L) + kron(L, I(N)) + kron(v1, v2)  # Full 2D matrix
    @time y_kron = H * x  # Compute using Kronecker product
    println("TEST Hmatvec: ", norm(vec(y_hmatvec') - y_kron) <= 1e-6)

    ### Exercise 8: Test CG_2 ###
    x_CG2 = CG_2(L, v1, v2, ones(N, N), 1e-6)  # Solve using CG_2
    res_CG2 = Hmatvec(L, v1, v2, x_CG2) - ones(N, N)  # Residual check
    println("TEST CG_2: ", norm(res_CG2) <= 1e-6)

    ### Exercise 9: Test deflation_2 ###
    eigenvalues_H = eigvals(H)  # Reference eigenvalues of matrix H
    jl_eigenvalues = sort(eigenvalues_H, by=abs)[1:5]  
    my_eigenvalues, my_eigenvectors = deflation_2(L, v1, v2, ones(N, N), 1e-6, 5)
    println("TEST deflation_2: ", norm(jl_eigenvalues - my_eigenvalues) <= 1e-6)
end

"""
Plots the convergence behavior of the Inverse Power Method.

This function analyzes and visualizes:
1. The convergence of the estimated eigenvalues.
2. The convergence of the iteratively computed eigenvectors.
3. The theoretical convergence rate based on the ratio of the two smallest eigenvalues.

The plot is saved as a PNG file in the current working directory.
"""
function plot_convergence()
    ### Data Initialization ###
    N = 1000  
    M = rand_schrodinger_1D(N)  
    Λ, Y = deflation(M, ones(N), 1e-8, 2) 

    # Analyze convergence of the Inverse Power Method
    Λ_it, Y_it = inverse_power_method_iterates(M, ones(N), 1e-8) 
    range = length(Λ_it)  

    ### Exercise 5: Plot Convergence ###
    
    # Eigenvalue convergence (absolute error)
    p = plot(1:range, abs.(Λ_it .- (Λ[1] * ones(range))),
             label=L"$|\lambda^{(m)} - \lambda_1|$", lw=2, xlabel=L"$m$", ylabel="Error",
             title="Convergence of Inverse Power Method (N=$N)")

    # Eigenvector convergence (norm difference to the final eigenvector)
    diff = zeros(range)
    for i in 1:range
        diff[i] = norm(Y_it[i] - Y_it[end])  # Difference to the last eigenvector
    end
    plot!(p, 1:range, diff, label=L"$||y^{(m)} - y_1||$", lw=2)

    # Theoretical convergence rate based on |λ1 / λ2|^m
    λ_ratio = abs(Λ[1] / Λ[2])  # Ratio of largest two eigenvalues
    theoretical_convergence = [λ_ratio^m for m in 1:range]  # Compute |λ1/λ2|^m
    plot!(p, 1:range, theoretical_convergence, label=L"$|\lambda_1 / \lambda_2|^m$", lw=2, ls=:dash)

    savefig(p, "Convergence_Inverse_Power_Method_N=$N.png")  
end

"""
Plots the relation between the normalized solution of the system and the eigenvectors.

This function:
1. Solves the linear system (L + v)x = 1 using the Conjugate Gradient method.
2. Computes the first five eigenvalues and eigenvectors of the matrix.
3. Compares the normalized solution x / ||x|| with the eigenvectors, highlighting
   the contribution of lower-energy levels.

The plot includes:
- The normalized solution x / ||x||.
- The first five eigenvectors with their corresponding eigenvalues.
"""
function plot_relation_1D()
	
    ### Data ###
    N = 1000
	M = rand_schrodinger_1D(N)
	
	### Exercise 2 : Solve ###
	x = CG(M, ones(N), 1e-15)

	### Exercise 6 : Compare ###
	Λ, Y = deflation(M, ones(N), 1e-15, 5)

    # Create the plot for the normalized solution x/||x||
    p = plot(1:N, x/norm(x), label="x/||x||")
    
    # Plot the 5 eigenvectors with their corresponding eigenvalues
    for i in 1:5 
        plot!(p, 1:N, Y[i, :], label=@sprintf("λ_%d = %.3f", i, Λ[i]))
    end
    
    # Customize the plot 
    xlabel!(p, "Index")
    ylabel!(p, "Value")
    title!(p, L"Solution and Eigenvectors of $(L + v)x = 1$")
    
    savefig(p, "Higher Impact of Lower Energy Levels (1D, N=$(N)).png")
end

"""
Generates 2D visualizations of the normalized solution and eigenvectors for varying grid sizes.

This function:
1. Solves the 2D Schrödinger equation for different grid sizes N using the CG method.
2. Computes the first five eigenvalues and eigenvectors using the deflation algorithm.
3. Plots the normalized solution x / ||x|| and the corresponding eigenvectors for comparison.

Each visualization is saved as a PNG file.

Grid sizes: [50, 100, 200]
"""
function plot_relation_2D()
	
    for N in [50, 100, 200]
        
        ### Data ###
        L = schrodinger_1D(N)
        v1 = potential_1D(N)
        v2 = potential_1D(N)

        ### Exercise 8: Solve ###   
        x = CG_2(L, v1, v2, ones(N,N), 1e-4) 
        plot_x = heatmap(x/norm(x), title="x/||x||", color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))

        ### Exercise 9: Eigenvalues ###   
        Λ, Y = deflation_2(L, v1, v2, ones(N,N), 1e-4, 5)

        title_λ1 = @sprintf("λ₁ = %.3f", Λ[1])
        title_λ2 = @sprintf("λ₂ = %.3f", Λ[2])
        title_λ3 = @sprintf("λ₃ = %.3f", Λ[3])
        title_λ4 = @sprintf("λ₄ = %.3f", Λ[4])
        title_λ5 = @sprintf("λ₅ = %.3f", Λ[5])

        nrows, ncols = size(Y[:,:,1])  # Hole die Dimensionen der Matrix
        plot_Y1 = heatmap(Y[:,:,1], title=title_λ1, color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))
        plot_Y2 = heatmap(Y[:,:,2], title=title_λ2, color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))
        plot_Y3 = heatmap(Y[:,:,3], title=title_λ3, color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))
        plot_Y4 = heatmap(Y[:,:,4], title=title_λ4, color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))
        plot_Y5 = heatmap(Y[:,:,5], title=title_λ5, color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))
      
        ### Exercise 10: Compare ###   
        p = plot(plot_x, plot_Y1, plot_Y2, plot_Y3, plot_Y4, plot_Y5) 
        savefig(p, "Higher Impact of Lower Energy Levels (2D, N=$(N)).png")
    end 
end

















