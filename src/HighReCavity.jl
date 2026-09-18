"""
HighReCavity — two-dimensional lid-driven cavity flow.  Chebyshev collocation on a mapped
Gauss–Lobatto grid, streamfunction representation ψ = (1−x²)(1−y²) q, Crank–Nicolson /
Adams–Bashforth time stepping; the implicit step is solved by an influence-matrix method built
on separable (Sylvester) solves.

    params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4)
    sim = CavitySimulation(params)
    run!(sim, 60_000; callback = s -> println(s.step), every = 1000)
    ψ = streamfunction(sim); ω = vorticity(sim); u, v = velocity(sim)
"""
module HighReCavity

using LinearAlgebra

include("chebyshev.jl")      # mapped Gauss–Lobatto grid, differentiation, interpolation
include("operators.jl")      # spectral operators of the q representation, physical fields
include("sylvester.jl")      # interior Dirichlet solves (Helmholtz, q-Poisson) as Sylvester equations
include("influence.jl")      # wall-node layout, influence matrix, solution of one implicit step
include("integrators.jl")    # time integrators (CNAB2, ARK3)
include("simulation.jl")     # parameters, state, step!, run!

export CavityParameters, CavitySimulation, step!, run!,
       streamfunction, vorticity, velocity, grid, reynolds_internal,
       ChebyshevGrid, CavityOperators, diff_matrices, interp_matrix,
       InfluenceSolver, influence_solve!, rhs!, laplacian!, laplacian_nodal!, biharmonic!, convection!

end # module
