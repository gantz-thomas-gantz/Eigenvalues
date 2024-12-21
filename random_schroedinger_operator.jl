using LinearAlgebra, Random


# #####################
# Utilities
# #####################
function cmp_vec(x,y,eps)
	if(length(x)!=length(y))
		return false
	else
		for i in 1:length(x)
			if(abs(x[i]-y[i])>eps)
				return false
			end
		end
	end
	return true
end

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

function invert(A)
    n = size(A, 1)
    A_inv = zeros(n, n)
    I = Diagonal(ones(n)) # Identity matrix

    for i in 1:n
        A_inv[:, i] = CG(A, I[:, i], 1e-20)
    end

    return A_inv
end

### Exercise 4 ###
function power_method(A, x, eps)
    y = x / norm(x)
    lambda = dot(y, A * y)  
    while norm(A * y - lambda * y) > eps
        y = A * y
        y = y / norm(y)
        lambda = dot(y, A * y)
    end
    return (lambda, y)
end



function Deflation(A::AbstractMatrix{T}, x::Vector{T}, eps::T, neig::Int) where T
    n_rows, n_cols = size(A)
    Λ = zeros(T, neig)  # Eigenvalue vector
    Y = zeros(T, neig, n_cols)  # Eigenvector matrix (one column per eigenvector)
    P = zeros(T, n_rows, n_cols)  # Initialize deflation matrix

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
        println(lambda)

		# Check if lambda is too small, skip deflation if so
		if abs(lambda) < eps
			println("Warning: Eigenvalue too small, skipping deflation.")
			continue
		end
        
        # Update the deflation matrix P using the outer product
        P += lambda * (y * y')  # Outer product to update deflation matrix
    end

    return (Λ, Y)
end


function DeflationX(A::AbstractMatrix, x::Vector, εtol::Float64, neig::Int)
    n_rows, n_cols = size(A)
    Λ = zeros(Float64, neig)  # Eigenvalue vector
    Y = zeros(Float64, neig, n_cols)  # Eigenvector matrix (one column per eigenvector)

    Pn_minus_1 = zeros(Float64, n_rows, n_cols)  # Initialize deflation matrix (non-square)
    
    for n in 1:neig
		
        # Compute the eigenvalue and eigenvector for the current iteration using power_method
        Λ[n], Y[n, :] = power_method(invert(A) - Pn_minus_1, x, 1e-40)

        # Update the deflation matrix Pn
        y = Y[n, :]
        Pn_minus_1 += 1/Λ[n] * (y * y')  # Outer product to update the deflation matrix
    end

    return (Λ, Y)
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
	x_jl = M\b
	my_x = CG(M,b,1e-6)
	println("TEST CG: ", cmp_vec(x_jl, my_x, 1e-6))

	### Exercise 3 : TEST Inverse Power Method ###
	jl_min_eigenvalue = eigenvalues[argmin(abs.(eigenvalues))]
	# (!) eigmin gives the not in absolute value smallest eigenvalue 
	my_l_min, my_y_min = inverse_power_method(M, ones(N), 1e-6) # random x
	println("TEST Inverse Power Method: ", abs(jl_min_eigenvalue - my_l_min) <= 1e-6)

	### Exercise 3: TEST Invert Matrix ###
    my_inv = invert(M)           # Your custom invert function
    jl_inv = inv(M)              # Julia's built-in matrix inversion
    println("TEST Invert Matrix: ", norm(my_inv - jl_inv) <= 1e-10)

	### Exercise 4: TEST Power method ###
	jl_max_eigenvalue = eigenvalues[argmax(abs.(eigenvalues))]
	my_l_max, my_y_max = power_method(M, ones(N), 1e-6) # random x
	println("TEST Power Method: ", abs(jl_max_eigenvalue - my_l_max) <= 1e-6)
	println(my_l_max)
	println(jl_max_eigenvalue)

	### Exercise 4: TEST Deflation Algorithm ###
	jl_eigenvalues = sort(eigenvalues, by=abs)[1:5]
	my_eigenvalues, my_eigenvectors = Deflation(M, ones(N), 1e-6, 5)
	println("jl: ", jl_eigenvalues)
	println("my: ", my_eigenvalues)
	println("TEST Deflation Algorithm: ", norm(jl_eigenvalues-my_eigenvalues) <= 1e-6)
	
end
















