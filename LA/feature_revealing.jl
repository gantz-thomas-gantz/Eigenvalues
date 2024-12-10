using MLDatasets
using Plots
using Statistics, LinearAlgebra


train_x, train_y = MNIST.traindata() #train_x 3D Array of the images, train_y = labels

# heatmap(train_x[:,:,1])


labels0 = findall(iszero,train_y[1:5000])
labels1 = findall(isone,train_y[1:5000])
labels = vcat(labels0,labels1)
A_new = train_x[:,:,labels]

for k=1:1042
	local m = mean(A_new[:,:,k])
        for i=1:28
               	for j=1:28
                   	A_new[i,j,k]-m
               	end
        end
end

global C = zeros(28*28)
for k=1:1042
	local v = reshape(A_new[:,:,k],28*28,1)
	hcat(C,v)
end

C=[1:end,2:end]

print(C)

u,s,v = svd(C)
p = scatter(v[labels0, 1],v[labels0, 2])
p = scatter!(v[labels1, 1],v[labels1, 2])
savefig(p, "feature_revealing.pdf")








