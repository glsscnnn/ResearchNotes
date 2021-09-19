using Flux

CNN = Chain(
   Conv((5, 5), 3 => 7),   # convolutional layer
   MaxPool((4, 4)),        # max pooling layer
   Flux.flatten,           # flatten
   Dense(7, 2)             # output layer
)

x = randn(Float32, 10, 10, 3, 20) # 20 10x10 RGB pictures
@show CNN(x)