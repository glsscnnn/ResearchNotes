using Flux

rNN = Chain(
   RNN(5, 5),   # RNNCell
   Dense(5, 1)  # Linear Output Layer
)

x = randn(Float32, 5, 5) |> Flux.flatten

@show rNN(x)