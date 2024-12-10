using LinearAlgebra, Plots

function Jacobi(A,b,x0)
	xk = x0 
	M = Diagonal(A)
	N = M - A
	it = 0
	while(norm((M*xk)-(N*xk+b))>10e-10)
		xk = inv(M)*(N*xk + b)
		it += 1
	end

	return xk, it
end

function GaussSeidel(A,b,x0)
	xk = x0 
	M = LowerTriangular(A)
	N = M - A
	it = 0
	while(norm((M*xk)-(N*xk+b))>10e-10)
		xk = inv(M)*(N*xk + b)
		it += 1
	end

	return xk, it
end

function error_evol_Jacobi(A,b,x0)
	
	x,it = Jacobi(A,b,x0)
	error = []
	spectral_radius = []
	
	xk = x0 
	M = Diagonal(A)
	N = M - A

	radius = maximum(broadcast(abs,eigvals(inv(M)*N)))
	
	for k in 1:it
		append!(error,norm(xk-x))
		append!(spectral_radius,radius^k)
		xk = inv(M)*(N*xk + b)
	end

	return 1:it, error, spectral_radius
end

function error_evol_GaussSeidel(A,b,x0)
	
	x,it = GaussSeidel(A,b,x0)
	error = []
	spectral_radius = []
	
	xk = x0 
	M = LowerTriangular(A)
	N = M - A

	radius = maximum(broadcast(abs,eigvals(inv(M)*N)))
	
	for k in 1:it
		append!(error,norm(xk-x))
		append!(spectral_radius,radius^k)
		xk = inv(M)*(N*xk + b)
	end

	return 1:it, error, spectral_radius
end

function test_error_evol()

	# initialize row-wise dominant matrix -> methods converge
	A = rand(3,3)
	for i in 1:3
		sum = 0
		for j  in 1:3
			sum += abs(A[i,j])
		end
		A[i,i] = sum
	end

	b = rand(3,1)
	x0 = rand(3,1)


	x_Jacobi, error_Jacobi, spectral_radius_Jacobi = error_evol_Jacobi(A,b,x0)
	x_GS, error_GS, spectral_radius_GS = error_evol_GaussSeidel(A,b,x0)
	
	p = plot(x_Jacobi,error_Jacobi,label="error")
	plot!(x_Jacobi,spectral_radius_Jacobi,label="spectral_radius")
	savefig(p,"Jacobi.pdf")
	
	p = plot(x_GS,error_GS,label="error")
	plot!(x_GS,spectral_radius_GS,label="spectral_radius")
	savefig(p,"GaussSeidel.pdf")
end


function test_random()

	# initialize row-wise dominant matrix -> methods converge
	A = rand(3,3)
	for i in 1:3
		sum = 0
		for j  in 1:3
			sum += abs(A[i,j])
		end
		A[i,i] = sum
	end

	b = rand(3,1)
	x0 = rand(3,1)

	println("real solution:")
	println(inv(A)*b)

	println("Jacobi:")
	solution_Jacobi, it_Jacobi = Jacobi(A,b,x0)
	println(solution_Jacobi, it_Jacobi)

	println("GaussSeidel:")
	solution_GaussSeidel, it_GaussSeidel = GaussSeidel(A,b,x0)
	println(solution_GaussSeidel, it_GaussSeidel)
end

function Lap_full(n)
	A = zeros(n,n)
	for i in 1:n
		A[i,i] = -2
		if i!=n
			A[i,i+1] = -1
		end
		if i!=1
			A[i,i-1] = -1
		end
	end
	return A
end
			


function Lap_sparse(n)
	A = Lap_full(n)
	return SymTridiagonal(A)
end

function test_Lap()

	Lap_f = Lap_full(100)
	Lap_s = Lap_sparse(100)
	b = rand(100,1)
	x0 = rand(100,1)

	
	println("## full Laplacian ##")
	
	println(typeof(Lap_f))

	println("Jacobi:")
	solution_Jacobi, it_Jacobi = @time Jacobi(Lap_f,b,x0)
	println(it_Jacobi)

	println("GaussSeidel:")
	solution_GaussSeidel, it_GaussSeidel = @time GaussSeidel(Lap_f,b,x0)
	println(it_GaussSeidel)

	println("## sparse Laplacian ##")
	
	println(typeof(Lap_s))

	println("Jacobi:")
	solution_Jacobi, it_Jacobi = @time Jacobi(Lap_s,b,x0)
	println(it_Jacobi)

	println("GaussSeidel:")
	solution_GaussSeidel, it_GaussSeidel = @time GaussSeidel(Lap_s,b,x0)
	println(it_GaussSeidel)
end









