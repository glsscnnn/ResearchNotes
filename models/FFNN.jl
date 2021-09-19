using Flux

FFNN = Chain(
    Dense(128, 64),
    Dense(64, 32),
    Dense(32, 32),
    Dense(32, 10)
)

x = randn(128, 1)
@show FFNN(x)
