# HighReCavity.jl

[![CI](https://github.com/uchytil/HighReCavity.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/uchytil/HighReCavity.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://uchytil.github.io/HighReCavity.jl/dev/)

A Chebyshev collocation solver for the two-dimensional lid-driven cavity at high Reynolds
number. The incompressible Navier–Stokes equations are solved in streamfunction–vorticity
form on the square cavity `[-1, 1]²`, driven by a regularised lid velocity `u(x, 1) = (1 − x²)²`,
with semi-implicit (Crank–Nicolson / Adams–Bashforth) time stepping. `Re` is the Reynolds
number based on the full cavity side length.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/uchytil/HighReCavity.jl")
```

## Usage

```julia
using HighReCavity

params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4)
sim = CavitySimulation(params)

run!(sim, 20_000)                                            # transient
run!(sim, 40_000; every = 200, callback = s -> process(s))   # process(sim) every 200 steps

ψ = streamfunction(sim); ω = vorticity(sim); u, v = velocity(sim)
```

`step!(sim)` advances a single step. The [documentation](https://uchytil.github.io/HighReCavity.jl/dev/)
describes the numerical method (mapped Chebyshev grid, the `ψ = (1−x²)(1−y²) q` representation,
the influence-matrix solution of the implicit step), the parameters and the post-processing
functions.

## License

MIT — see [LICENSE](LICENSE).
