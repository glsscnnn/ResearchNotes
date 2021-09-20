using Flux
using NNlib

mutable struct RBM
   visible::Flux.Chain
   hidden::Flux.Chain
end

# layer one is a RBM that has a hidden and visible layer
h1 = RBM(Chain(
   Dense(4, 4),
   Dense(4, 4),
   Dense(4, 4)
),
Chain(
   Dense(4, 4),
   Dense(4, 4),
   Dense(4, 4)
))

x = randn(Float32, 4)                              # fake data
x = softmax((h1.hidden ∘ h1.visible)(x), dims=1)   # softmax

# layer two is the network that we pass our RBM to
h2 = Chain(
   Dense(4, 4),
   Dense(4, 2)
)

"""
Training of a DBN is done differently this is not really
illustrated here that well :(
"""

@show h2(x)