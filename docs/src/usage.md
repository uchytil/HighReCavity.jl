# [Usage](@id usage)

## Usage

```julia
params = CavityParameters(
    N       = 128,      # polynomial degree, (N+1)² nodes
    Re      = 30_000,   # Reynolds number, full side length L = 2
    dt      = 5e-4,     # time step
    alpha   = 0.96,     # grid mapping parameter (0 = unmapped)
    backend = :ceigen,  # :ceigen or :schur
    integrator = :ark3, # :ark3 (default) or :cnab2
)
sim = CavitySimulation(params)   # grid, operators, decompositions, influence matrix (fixed cost)

step!(sim)          # one time step
run!(sim, 60_000)   # many steps
sim.step, sim.t     # steps taken, current time
```

`CavityParameters` also accepts `T = Float32` etc. to choose the floating-point type. With the
default backend and integrator, initialisation takes about 0.4 s at N = 128 and 4 s at N = 256, and
one step about 5 ms and 32 ms respectively (Apple M2 Max, 8 BLAS threads); a CNAB2 step costs
about a third of that but requires a ~3× smaller step.

## Post-processing

```julia
sim.q                 # state q,  ψ = (1−x²)(1−y²) q
streamfunction(sim)   # Ψ = W q W
vorticity(sim)        # ω = Δψ
velocity(sim)         # (u, v) = (ψ_y, −ψ_x)
grid(sim).x           # node coordinates (same in x and y)
```

All fields are `(N+1)×(N+1)` arrays indexed `[i, j] ↔ (x_i, y_j)`. `sim.ω` holds the vorticity
produced by the last implicit solve, which equals `vorticity(sim)`. The same functions accept
`(q, ops)` directly, e.g. `velocity(q, sim.ops)`, for states saved earlier. To evaluate a field
on a uniform grid for plotting:

```julia
xs = range(-1, 1, length = 201)
M  = interp_matrix(collect(xs), grid(sim))     # barycentric interpolation matrix
Ψ_plot = M * streamfunction(sim) * M'
```

## Long simulations

`run!` keeps only the current state. Periodic output goes through the callback, which is
called with the simulation after every `every`-th step (and after the last one):

```julia
run!(sim, 20_000)                                          # transient, nothing recorded

using Serialization
energy = Float64[]
function collect!(s)
    u, v = velocity(s)
    push!(energy, sum(abs2, u) + sum(abs2, v))
    s.step % 5000 == 0 && serialize("q_$(s.step).jls", (q = s.q, q_prev = s.q_prev, t = s.t))
end
run!(sim, 1_000_000; every = 200, callback = collect!)      # production interval
```

`run!` stops with an error if the solution becomes NaN (`check_nan = false` disables the
check). A run can be resumed from a saved `(q, q_prev)` pair by copying them into `sim.q` and
`sim.q_prev` of a simulation built with the same parameters. See `examples/run_cavity.jl`.


## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'` checks the implicit solve against a direct
dense solution of the same discrete equations at small ``N``, short trajectories against stored
reference data, the boundary conditions, and the agreement of the two backends.
