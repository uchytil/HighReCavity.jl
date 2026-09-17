# HighReCavity.jl

A Chebyshev collocation solver for the two-dimensional lid-driven cavity at high Reynolds
number, in streamfunction–vorticity form with semi-implicit (Crank–Nicolson /
Adams–Bashforth) time stepping.

The flow in the square cavity ``[-1,1]^2`` is driven by the top wall moving with the
regularised velocity ``u(x,1) = (1-x^2)^2``; the other walls are at rest. The Reynolds number
``Re = UL/\nu`` is based on the full side length ``L = 2`` and the peak lid speed ``U = 1``.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/uchytil/HighReCavity.jl")
```

## Quick start

```julia
using HighReCavity

params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4)
sim = CavitySimulation(params)

run!(sim, 20_000)                                       # transient
run!(sim, 40_000; every = 200, callback = s -> nothing) # production interval, called every 200 steps

ψ = streamfunction(sim); ω = vorticity(sim); u, v = velocity(sim)
```

- [Numerical method](@ref method): equations, discretisation and the implicit solve.
- [Usage](@ref usage): parameters, running, post-processing, long simulations.
- [API](@ref api).
