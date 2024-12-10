using LinearAlgebra

function Arnoldi(A,v,k)
	n = length(A[1,:])
	V = zeros(n,k+1)
	H = zeros(k+1,k)
	V[:,1] = v/norm(v)
	for j in 1:k
		for i in 1:j
			H[i,j] = dot(V[:,i],A*V[:,j])
		end
		sum = zeros(length(v))
		for i in 1:j
			sum += H[i,j]*V[:,i]
		end
		v_hat = A*V[:,j] - sum
		H[j+1,j] = norm(v_hat)
		if H[j+1,j]!=0
			V[:,j+1] = v_hat/H[j+1,j]
		end
	end
	return V,UpperHessenberg(H)
end

function GMRES(A,b,x0,K)
	r0 = b-(A*x0)
	V,H = Arnoldi(A,r0,K)
	q,r = qr(H)
	
	e1 = zeros(K+1,1)
	e1[1] = 1
	R = norm(r0)*transpose(q)*e1
	t = inv(r)*R[1:K]
	return x0+V[:,1:K]*t, abs(R[K+1])
end

function T(n, alpha,sigma)
	return Tridiagonal(-alpha*ones(n-1),(sigma+alpha)*ones(n),(-1/alpha)*ones(n-1))
end

function test(n,K_max)
	residuals = zeros(K_max)
	Ks = zeros(K_max)
	for K in 1:K_max
		x, r = GMRES(T(n,0.9,2.0), rand(n,1), zeros(n), K)
		residuals[K] = r
		Ks[K] = K
	end
	p = plot(Ks[50:end],residuals[50:end])
	savefig(p,"GMRES.pdf")
end
