using LinearAlgebra, Plots

function Lap_full(n)
          A = zeros(n,n)
	  for i in 1:n
                  A[i,i] = 2
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

function steepest_gradient(A,b,eps)
	x = zeros(length(b))
	p = b
	while(norm(p)>eps)
		alpha = dot(p,p)/dot(p,A*p)
		x += alpha*p
		p -= alpha*A*p
	end
	return x
end

function conjugate_gradient(A,b,eps)
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

function Anorm(A,x,y)
	return sqrt(dot(x-y,A*(x-y)))
end

function steepest_gradient_error(A,b,eps,sol)
	x = zeros(length(b))
	p = b
	VecErrors = [Anorm(A,x,sol)]
	while(norm(p)>eps)
		alpha = dot(p,p)/dot(p,A*p)
		x += alpha*p
		p -= alpha*A*p
		append!(VecErrors,Anorm(A,x,sol))
	end
	return VecErrors
end

function conjugate_gradient_error(A,b,eps,sol)
	x = zeros(length(b))
	p = b - A*x
	rk_1 = p
	rk = p
	VecErrors = [Anorm(A,x,sol)]
	while(norm(rk)>eps)
		alpha = dot(rk_1,rk_1)/dot(p,A*p)
		x += alpha*p
		rk = rk_1 - alpha*A*p
		omega = dot(rk,rk)/dot(rk_1,rk_1)
		rk_1 = rk
		p = rk + omega*p
		append!(VecErrors,Anorm(A,x,sol))	
	end
	return VecErrors
end
function test(n)
	T = Lap_sparse(n)
	b = rand(n,1)
	println("# steepest gradient:")
	@time steepest_gradient(T,b,1e-6)
	println("# conjugate gradient:")
	@time conjugate_gradient(T,b,1e-6)
end

function plot_error(n)
	eps = 1e-6
	T = Lap_sparse(n)
	b = rand(n,1)
	sol = T\b
	
	VecErrorSG = steepest_gradient_error(T,b,eps,sol)
	VecErrorCG = conjugate_gradient_error(T,b,eps,sol)

	K_SG = [i for i in range(0,length(VecErrorSG)-1)]
	K_CG = [i for i in range(0,length(VecErrorCG)-1)]
	
	lamda_n = eigmax(T)
	lamda_1 = eigmin(T)
	kappa = lamda_n/lamda_1
		
	val_SG = (kappa-1)/(kappa+1)
	BoundSG = [(val_SG^k)*Anorm(T,zeros(n),sol) for k in K_SG]

	val_CG = (sqrt(kappa)-1)/(sqrt(kappa)+1)
	BoundCG = [2*(val_CG^k)*Anorm(T,zeros(n),sol) for k in K_CG]

	p = plot(K_SG,VecErrorSG,label="error")
	plot!(K_SG,BoundSG,label="bound")
	savefig(p,"steepest_gradient.pdf")

	start = 3000
	p = plot(K_CG,VecErrorCG,label="error")
	plot!(K_CG,BoundCG,label="bound")
	savefig(p,"conjugate_gradient.pdf")

end
		
	

