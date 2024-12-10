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
	p = b - A*x
	rk_1 = p
	rk = p
	while(norm(rk)>eps)
		alpha = dot(rk_1,rk_1)/dot(p,A*p)
		x += alpha*p
		rk = rk_1 - alpha*A*p
		omega = dot(rk,rk)/dot(rk_1,rk_1)
		rk_1 = rk
		p = rk + omega*p
	end
	return x
end

# actually does not need to invert A bust solves systems instead
function inverse_power_method(A, x, eps)
	y = x/norm(x)
	x = CG(A,y,1e-8)
	l = dot(y,x)
	while(norm(y-l*A*y)>eps)
		y = x/norm(x)
		x = CG(A,y,1e-8)
		l = dot(y,x)
	end
	return 1/l, y
end

# TODO: write invert function by CG

### Exercise 4 ###
# This one needs to invert A
function deflation(A, x, eps, n_eig)
	L = zeros(n_eig)
	Y = zeros(n_eig,length(x)) # (!) eigenvectors are stored in the columns
	for n in 1:n_eig
		sum = 0
		for i in 1:n-1 
			sum += 1/L[i]*Y[i]*adjoint(Y[i])
		end
		L[n], Y[n,:] = inverse_power_method(A_inv, ones(length(x)), eps)
	end
	return L, Y
end




# #####################
# MAIN
# #####################
function main()
	
	N = 1000
	M = rand_schrodinger_1D(N)
	
	### Exercise 2/3 : TEST CG ###
	b = ones(N)
	x_jl = M\b
	my_x = CG(M,b,1e-6)
	println("TEST CG: ", cmp_vec(x_jl, my_x, 1e-6))

	### Exercise 3 : TEST Inverse Power Method ###
	l_min_jl = minimum(abs.(eigvals(M))) # (!) eigmin gives the not in absolute value smallest eigenvalue 
	my_l_min, my_y_min = inverse_power_method(M, ones(N), 1e-6) # random x
	println("TEST Inverse Power Method: ", abs(l_min_jl - my_l_min) <= 1e-6)


end















