using Flux

# encoding layer
encoder = Chain(
   Dense(5, 4),
   Dense(4, 4),
   Dense(4, 2)
)

# decoding layer
decoder = Chain(
   Dense(2, 4),
   Dense(4, 4),
   Dense(4, 5)
)

# fake data
x = randn(5)

# thanks julia
@show (decoder ∘ encoder)(x)