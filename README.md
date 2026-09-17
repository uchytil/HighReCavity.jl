# HighReCavity

Influence-matrix (streamfunction–vorticity) reformulation of the 2-D lid-driven-cavity Chebyshev
collocation solver `reference/cavity-flow-spectral-ab2-cn.jl` (kept verbatim).  The dense
`(N+1)²×(N+1)²` inverse of the reference is replaced by 1-D decompositions plus an `m×m`
boundary system (`m = 4(N−1)`), reproducing the reference to roundoff.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                     # 401 tests
julia --project=. scripts/compare_reference.jl 128               # validation vs A⁻¹ (N = 32, 64, 128)
julia --project=. scripts/diagnostics_small_N.jl                 # exact algebra at N = 8
julia --project=. scripts/run_cavity.jl 128 60000 ceigen         # the reference run, new solver

# benchmarks use the separate `bench` environment (BenchmarkTools); once:
julia --project=bench -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=bench bench/benchmark.jl 128                     # per-step benchmark
julia --project=bench bench/scaling.jl                           # N = 32 … 256
```

Minimal use:

```julia
using HighReCavity
grid = ChebyshevGrid((128, 128), (0.96, 0.96), Float64)
ops  = CavityOperators(Matrix{Float64}, grid)
solver = InfluenceSolver(QForm, grid, ops, 0.0005, 15000.0; poisson_mode = :ceigen)  # :schur = default
st = QState(solver)                       # q with lid data, q_prev = 0 (as the reference)
for n in 1:1000
    step_influence!(st, solver)           # st.q, st.q_prev, st.ω
end
ψ = streamfunction_from_q(st.q, ops); u, v = velocity(st.q, ops)
```

See `notes/technical_note.md` for the mathematics (exact equivalence with the reference,
corner/rank analysis, why the textbook nodal ψ–ω formulation is a different discretization) and the
measured performance.
