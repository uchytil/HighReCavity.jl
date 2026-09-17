"""
HighReCavity — 2-D lid-driven cavity, Chebyshev collocation on a mapped grid, streamfunction
ψ = (1−x²)(1−y²) q, AB2/Crank–Nicolson time stepping with an influence-matrix solve of the
implicit step (no dense (N+1)²×(N+1)² operator).

    params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4)     # Re: full side length L = 2
    sim = CavitySimulation(params)
    run!(sim, 60_000; callback = s -> println(s.step), every = 1000)
    ψ = streamfunction(sim); ω = vorticity(sim); u, v = velocity(sim)
"""
module HighReCavity

using LinearAlgebra

include("chebyshev.jl")      # grid, differentiation, interpolation
include("operators.jl")      # q-formulation operators, RHS building blocks, physical fields
include("sylvester.jl")      # interior Dirichlet solves (Helmholtz, q-Poisson) via 1-D decompositions
include("influence.jl")      # boundary layout, influence matrix, one implicit solve
include("simulation.jl")     # parameters, state, step!, run!

export CavityParameters, CavitySimulation, step!, run!,
       streamfunction, vorticity, velocity, grid, reynolds_internal,
       ChebyshevGrid, CavityOperators, diff_matrices, interp_matrix,
       InfluenceSolver, influence_solve!, rhs!, laplacian!, biharmonic!, convection!

end # module
