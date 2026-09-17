using HighReCavity
using Test, LinearAlgebra, Random, Printf, Logging

const Δt_test = 0.0005
const Re_test = 15000.0

# small-N helper
function setup(N; α = 0.96)
    grid = ChebyshevGrid((N, N), (α, α), Float64)
    ops = CavityOperators(Matrix{Float64}, grid)
    return grid, ops
end
smooth_q(grid) = [exp(-(x^2 + y^2)) * sin(2x) * cos(1.3y) + 0.3x * y^2 for x in grid.xs[1], y in grid.xs[2]]

@testset "HighReCavity" begin
    include("test_reference_relations.jl")
    include("test_psiomega.jl")
    include("test_sylvester.jl")
    include("test_boundary.jl")
    include("test_influence.jl")
    include("test_timestep.jl")
end
