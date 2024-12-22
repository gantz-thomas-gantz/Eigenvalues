# TODO: 5., 6.
# TODO: 7. (x,y are matrices ->heatmap with strongness is their value)
# TODO: 8. (CG auf Matrizen ?)


using LinearAlgebra, Random

# #####################
# One-dimensional case
# #####################

### Exercise 1 ###
function rand_schrodinger_1D(N)
	
	# L
	dv = 2  * ones(N)
	ev = -1 * ones(N-1)
	L  = SymTridiagonal(dv, ev)

	# v
	for i in 1:N
		dv[i] = rand((1.0,1/N^2))
	end
	v = Diagonal(dv)
	
	return L + v
end

function schrodinger_1D(N)
	
	# L
	dv = 2  * ones(N)
	ev = -1 * ones(N-1)
	L  = SymTridiagonal(dv, ev)
	
	return L 
end

function potential_1D(N)

    # v
    dv = zeros(N)
	for i in 1:N
		dv[i] = rand((1.0,1/N^2))
	end
	v = Diagonal(dv)
	
	return v
end



### Exercise 3 ###
function CG(A, b, eps)
    x = zeros(length(b))
    p = b - A * x
    rk_1 = p
    rk = p

    while norm(rk) > eps
        alpha = dot(rk_1, rk_1) / dot(p, A * p)
        x += alpha * p
        rk = rk_1 - alpha * A * p
        omega = dot(rk, rk) / dot(rk_1, rk_1)
        rk_1 = rk
        p = rk + omega * p
    end

    return x
end

function inverse_power_method(A, x, eps)
    y = x / norm(x)
    x = CG(A, y, 1e-20)
    l = dot(y, x)

    while norm(y - l * A * y) > eps
        y = x / norm(x)
        x = CG(A, y, 1e-20) # no matrix inversion but solving a system
        l = dot(y, x)
    end

    return 1 / l, y
end

### Exercise 4 ###
function deflation(A, x, eps, neig)
    n_rows, n_cols = size(A)
    Λ = zeros(neig)             # Eigenvalue vector
    Y = zeros(neig, n_cols)     # Eigenvector matrix (one row per eigenvector)
    P = zeros(n_rows, n_cols)   # Initialize deflation matrix

    for n in 1:neig
        # Initial vector normalization
        y = x / norm(x)
        
        # Solve for v using the CG method with more reasonable tolerance
        v = CG(A, y - A * P * y, 1e-8)
        
        # Compute initial eigenvalue estimate (Rayleigh quotient)
        lambda = dot(y, v)
        
        # Convergence loop with relative tolerance check
        while norm(v - lambda * y) / norm(v) > eps
            y = v / norm(v)  # Re-normalize y
            v = CG(A, y - A * P * y, 1e-8)
            lambda = dot(y, v)
        end

        # Store eigenvalue and eigenvector
        Λ[n], Y[n, :] = 1 / lambda, y
        
        # Update the deflation matrix P using the outer product
        P += lambda * (y * y')  # Outer product to update deflation matrix
    end

    return (Λ, Y)
end

# #####################
# Two-dimensional case
# #####################

### Exercise 7 ###
function Hmatvec(L, v, w, x)
    N = size(L,1)
    
    y = L*x + x*L
 
    for j1 in 1:N
        for j2 in 1:N 
                y[j1,j2] += v[j1,j1]*w[j2,j2]*x[j1,j2]
        end
    end

    return y
end



# #####################
# MAIN
# #####################
function main()
	
    N = 50
	M = rand_schrodinger_1D(N)
	eigenvalues = eigvals(M) 
	
	### Exercise 2 : TEST CG ###
	b = ones(N)
	jl_x = M\b
	my_x = CG(M,b,1e-6)
	println("TEST CG: ",  norm(jl_x - my_x) <= 1e-6)

	### Exercise 3 : TEST Inverse Power Method ###
	jl_min_eigenvalue = eigenvalues[argmin(abs.(eigenvalues))] # (!) eigmin gives the not in absolute value smallest eigenvalue 
	my_l_min, my_y_min = inverse_power_method(M, ones(N), 1e-6) # random x
	println("TEST Inverse Power Method: ", abs(jl_min_eigenvalue - my_l_min) <= 1e-6)

	### Exercise 4: TEST Deflation Algorithm ###
	jl_eigenvalues = sort(eigenvalues, by=abs)[1:5]
	my_eigenvalues, my_eigenvectors = deflation(M, ones(N), 1e-6, 5)
	println("TEST Deflation Algorithm: ", norm(jl_eigenvalues-my_eigenvalues) <= 1e-6)
	
    ### Exercise 7: TEST Hmatvec ###
    x = rand(N * N)
    x_matrix = reshape(x, N, N)'
    L = schrodinger_1D(N)
    v1 = potential_1D(N)
    v2 = potential_1D(N)
    @time y_hmatvec = Hmatvec(L, v1, v2, x_matrix) 
    H = kron(I(N), L) + kron(L, I(N)) + kron(v1, v2)
    @time y_kron = H * x
    println("TEST Hmatvec: ", norm(vec(y_hmatvec') - y_kron) <= 1e-6)

end
















