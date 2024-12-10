using LinearAlgebra

function printMatrix(A)
	n = size(A)[1]
	for i = 1:n
		println(A[i,:])
	end
end

function lu_naive(A)
	n = size(A)[1]
	L = Matrix{Float64}(I,n,n)
	U = deepcopy(A)
	for p = 1:n-1
		for i = p+1:n
			L[i,p] = U[i,p]/U[p,p]
			for j = 1:n
				U[i,j] -= L[i,p]*U[p,j]
			end
		end
	end
	return L,U
end


function lu_decompose(A)
	n = size(A)[1]
	T = deepcopy(A)
	for p = 1:n-1
		for i = p+1:n
			T[i,p] = T[i,p]/T[p,p]
			for j = p+1:n
				T[i,j] -= T[i,p]*T[p,j]
			end
		end
	end
	return T
end

function lu_solve(A,b)
	
	T = lu_decompose(A)
	n = size(A)[1]
	
	v = zeros(size(b))
	# lower triangular solve Lv = b
	for i = 1:n 
		v[i] = b[i]
		for j = 1:i-1
			v[i] -= T[i,j]*v[j]
		end
	end


	u = zeros(size(b))
	# upper triangular solve Uu = v
	for p = 1:n 
		i = n-p+1
		u[i] = v[i]
		for j = i+1:n
			u[i] -= T[i,j]*u[j]
		end
		u[i] = u[i]/T[i,i]
	end

	return u
end




function examples_test()
	A = ones(3,3)
	A[2,2] = 2
	A[3,3] = 10
	println("A:")
	printMatrix(A)
	println("")
	
	println("################")
	println("lu_naive:")
	L,U = @time lu_naive(A)

	println("L:")
	printMatrix(L)
	println("")

	println("U:")
	printMatrix(U)
	println("")
	
	println("################")
	println("lu_decompose:")
	T = @time lu_decompose(A)

	println("T:")
	printMatrix(T)
	println("")

	println("################")
	println("lu_solve:")
	b = ones(3)
	b[2] = 10
	println("A:")
	printMatrix(A)
	println("")
	println("b:")
	printMatrix(b)
	println("")
	
	u = @time lu_solve(A,b)

	println("u:")
	printMatrix(u)
	println("")
end

function time_test()

	steps = 10
	Time = zeros(steps)

	for k=1:steps
		N = 2^k
		A = ones(N,N)
		for i = 1:N
			A[i,i] = N
		end
		b = ones(N)
		
		t = time()
		lu_solve(A,b)
		Time[k] = time()-t
	end
	printMatrix(Time)
end




