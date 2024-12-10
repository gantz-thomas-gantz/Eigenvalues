using FileIO, Images, LinearAlgebra, Plots

img_file = load("jura.jpg") 
img_data = channelview(img_file) 
	
function f()
	u,s,v = svd(img_data[1,:,:])
	p = plot(1:1:length(s),s,yaxis=:log)

	for i = 2:3
		u,s,v = svd(img_data[i,:,:])
		plot!(1:1:length(s),s,yaxis=:log)
	end
	
	savefig(p,"singular_values.pdf")
end

function rank_approximation(A,r)
	u,s,v = svd(A)
	return u[:,1:r], s[1:r], v[:,1:r]
end

function g()
	
	#for i = 1:3
		u,s,v = svd(img_data[1,:,:])
	#end
			
	rank = 1:1:10
	error = zeros(10)
	
	for i = 1:10
		error[i] = norm(s[(rank[i]+1):end],2) # Poly Remark 1.20
	end
	println(error)
	p = plot(rank,error)
	savefig(p,"rank_error.pdf")
end

function h()
	out = zeros(size(img_data))
	for i = 1:3
		u,s,v = rank_approximation(img_data[i,:,:],20)
		out[i,:,:] = u*Diagonal(s)*v'
	end

	img_approx_file = colorview(RGB,clamp01.(out)) 
	save("approx.png",img_approx_file) 
end
	
	
	
	
	
	
	
	
	
		
