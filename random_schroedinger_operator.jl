using LinearAlgebra, Random
using Plots, LaTeXStrings, Printf

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
    x = CG(A, y, 1e-8)
    l = dot(y, x)

    while norm(y - l * A * y) > eps
        y = x / norm(x)
        x = CG(A, y, 1e-8) # no matrix inversion but solving a system
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

### Exercise 5 ###
function inverse_power_method_iterates(A, x, eps)
    Λ, Y = [], []
    
    y = x / norm(x)
    x = CG(A, y, 1e-8)
    l = dot(y, x)

    append!(Λ,1/l) 
    push!(Y, y)     

    while norm(y - l * A * y) > eps
        y = x / norm(x)
        x = CG(A, y, 1e-8) # no matrix inversion but solving a system
        l = dot(y, x)

        append!(Λ,1/l) 
        push!(Y, y)   
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

### Exercise 8 ###
function CG_2(L, v, w, b, eps)
    x = zeros(size(b))
    p = b - Hmatvec(L, v, w, x)
    rk_1 = p
    rk = p

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
function deflation_2(L, v1, v2, x, eps, neig)
    N = size(L, 1)
    Λ = zeros(neig)             # Eigenvalue vector
    Y = zeros(size(x)..., neig)
    P = zeros(N*N,N*N)

    for n in 1:neig
        println(n)
        # Initial vector normalization
        y = x / norm(x)
        
        # Solve for v using the CG method with more reasonable tolerance
        Py = reshape(P * vec(y'), N, N)'
        v = CG_2(L, v1, v2, y - Hmatvec(L, v1, v2, Py), 1e-8)
        
        # Compute initial eigenvalue estimate (Rayleigh quotient)
        lambda = dot(vec(y'),vec(v'))
        
        # Convergence loop with relative tolerance check
        while norm(v - lambda * y) / norm(v) > eps
            y = v / norm(v)  # Re-normalize y
            Py = reshape(P * vec(y'), N, N)'
            v = CG_2(L, v1, v2, y - Hmatvec(L, v1, v2, Py), 1e-8)
            lambda = dot(vec(y'),vec(v'))
        end

        # Store eigenvalue and eigenvector
        Λ[n], Y[:,:,n] = 1 / lambda, y
        
        # Update the deflation matrix P using the outer product
        P += lambda * (vec(y') * vec(y')')  # Outer product to update deflation matrix
    end

    return (Λ, Y)
end



# #####################
# MAIN
# #####################
function test()
	
    N = 20
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

    ### Exercise 8: TEST CG_2 ###   
    println("TEST CG_2: ", norm( ones(N,N) - Hmatvec(L, v1, v2, CG_2(L, v1, v2, ones(N,N), 1e-6) ) ) <= 1e-6)

    ### Exercise 9: TEST deflation_2 ###   
    eigenvalues_H = eigvals(H)
    jl_eigenvalues = sort(eigenvalues_H, by=abs)[1:5]
    my_eigenvalues, my_eigenvectors = deflation_2(L, v1, v2, ones(N,N), 1e-6, 5)
    println("TEST deflation_2: ", norm(jl_eigenvalues-my_eigenvalues) <= 1e-6)
  
end

function plot_convergence()
	
    ### Data ###
    N = 1000
	M = rand_schrodinger_1D(N)
    Λ, Y = deflation(M, ones(N), 1e-6, 2)
	
	### Exercise 5 : Analyse convergence ###
	Λ_it, Y_it = inverse_power_method_iterates(M, ones(N), 1e-6)
    range = length(Λ_it)
    
    # Eigenvalues
    p = plot(1:range, abs.(Λ_it .- (Λ[1] * ones(range))), label=L"$|\lambda^{(m)} - \lambda_1|$")
    
    # Eigenvectors
    diff = zeros(range)
    for i in 1:range
        diff[i] = norm(Y_it[i] - Y_it[range])
    end  
    plot!(p, 1:range, diff, label=L"$||y^{(m)} - y_1||$")
    
    a = abs((Λ[1] / Λ[2])) * ones(range)
    b = [a[i]^(i) for i in 1:range]
    plot!(p, 1:range, b, label=L"$|\lambda_1 / \lambda_2|^m$")
    
    xlabel!(p, L"$m$")
    ylabel!(p, "Error")
    title!(p, "Convergence Inverse Power Method (N=$(N))") 
    savefig(p, "Convergence Inverse Power Method (N=$(N)).png")

end

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
    for i in 1:5  # Loop variable changed to I
        plot!(p, 1:N, Y[i, :], label=@sprintf("λ_%d = %.3f", i, Λ[i]))
    end
    
    # Customize the plot with labels, title, and legend
    xlabel!(p, "Index")
    ylabel!(p, "Value")
    title!(p, L"Solution and Eigenvectors of $(L + v)x = 1$")
    
    # Save the plot as an image file
    savefig(p, "Higher Impact of Lower Energy Levels (1D, N=$(N)).png")
end

function plot_relation_2D()
	
    for N in [50] 
        
        # , 100, 200]
        
        ### Data ###
        L = schrodinger_1D(N)
        v1 = potential_1D(N)
        v2 = potential_1D(N)

        ### Exercise 8: Solve ###   
        x = CG_2(L, v1, v2, ones(N,N), 1e-6) 
        plot_x = heatmap(x/norm(x), title="x/||x||", color=:viridis, xticks=([1, N], ["1", string(N)]), yticks=([1, N], ["1", string(N)]))

        ### Exercise 9: Eigenvalues ###   
        Λ, Y = deflation_2(L, v1, v2, ones(N,N), 1e-6, 5)

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
















