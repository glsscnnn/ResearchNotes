using Flux

# Generator
Gen = Chain(
   ConvTranspose((4, 4), 3 => 32, relu),
   BatchNorm(32),
   ConvTranspose((4, 4), 32 => 16, relu),
   ConvTranspose((4, 4), 16 => 8 , relu),
   BatchNorm(8),
   ConvTranspose((4, 4), 8 => 3, hardtanh)
)

# Discriminator
Dis = Chain(
   Conv((8, 8), 3 => 32, leakyrelu),
   Conv((8, 8), 32 => 16, leakyrelu),
   BatchNorm(16),
   Conv((8, 8), 16 => 3, sigmoid),
)
