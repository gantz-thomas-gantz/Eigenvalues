function log_map(x_0, r, N)
	
	list = [x_0]
	
	for i in 1:(N-1)
		append!(list, r*list[i]*(1-list[i]))
	end

	return list


end

function plot2()
	
	X = 1:1:100
	Y = log_map(0.5, 0.5, 100)
	p = plot(X,Y,label="r=0.5")
	
	Y = log_map(0.5, 1, 100)
	plot!(X,Y,label="r=1")
	
	Y = log_map(0.5, 2, 100)
	plot!(X,Y,label="r=2")
	
	Y = log_map(0.5, 3, 100)
	plot!(X,Y,label="r=3")
	
	Y = log_map(0.5, 3.9, 100)
	plot!(X,Y,label="r=3.9")
	
	Y = log_map(0.5, 4, 100)
	plot!(X,Y,label="r=4")
	
	savefig(p,"plot.pdf")

end


function plot3()

	X = []
	R = 0:0.005:4

	for r in R
		append!(X, log_map(0.5, r, 200)[200])
	end

	p = scatter(X)

	for N in 205:5:300
		X = []


		for r in R
			append!(X, log_map(0.5, r, N)[N])
		end

		p = scatter!(X)
	end

	savefig(p, "plot3.pdf")
end

