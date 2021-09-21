# less abstraction RBM
# still want flux for util tho
using Flux

mutable struct RBM
   hidden
   visible
end

# TODO write scratch
prod(sigmoid(a + sum(weights, hidden)))