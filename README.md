# HighReCavity.jl

2-D lid-driven cavity at high Reynolds number: Chebyshev collocation on an arcsin-mapped grid,
streamfunction `ψ = (1−x²)(1−y²) q`, AB2/Crank–Nicolson time stepping. The implicit step is
solved with an influence matrix built from 1-D operator decompositions, so no dense
`(N+1)²×(N+1)²` operator is ever formed (memory O(N²), ~1–4 ms per step at N = 128 on a laptop).

The numerics are those of the original dense-inverse solver (`reference/`), reproduced to
roundoff; see `docs/validation_note.md` and the tag `v0.1.0-validated` for the full validation.

```julia
using HighReCavity

params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4, alpha = 0.96, backend = :ceigen)
sim = CavitySimulation(params)          # grid, operators, decompositions, influence matrix

run!(sim, 20_000)                                                # transient, nothing stored
run!(sim, 40_000; every = 200, callback = s -> process(s))      # production interval
step!(sim)                                                       # single step

sim.q                       # state, ψ = (1−x²)(1−y²) q
streamfunction(sim)         # Ψ = W q W
vorticity(sim)              # ω = Δψ (exact product-rule operators)
velocity(sim)               # (u, v) = (ψ_y, −ψ_x)
```

**Reynolds number.** `Re` is based on the full cavity side length `L = 2` (cavity `[-1,1]²`,
lid speed `max (1−x²)² = 1`). The operators are written in half-width units, so the solver uses
`Re_internal = Re/2` internally (`reynolds_internal(params)`); `Re = 30_000` reproduces the
original script's `Re = 30000/2`.

**Backends.** `:ceigen` (default): complex eigendecompositions, 4 GEMMs per Sylvester solve.
`:schur`: real Schur + LAPACK `trsyl`, backward stable, ~4× slower; useful as a cross-check.

Layout: `src/chebyshev.jl` (grid, differentiation), `src/operators.jl` (q-operators, fields),
`src/sylvester.jl` (Helmholtz / q-Poisson interior solves), `src/influence.jl` (influence
matrix, one implicit solve), `src/simulation.jl` (parameters, `step!`, `run!`).
Tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (dense equivalence at N = 8/12, trajectories
vs the preserved implementation, boundary conditions, backend agreement, Re convention).

## License

MIT — see [LICENSE](LICENSE).
