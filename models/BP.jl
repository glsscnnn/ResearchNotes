using Flux

include("FFNN.jl")

loss = Flux.mse            # loss function
datapoints = randn(4, 4)   # fake data
opt = Descent(0.1)         # gradient descent optmizer

# TODO BP