# Influence-matrix reformulation of the Chebyshev lid-driven-cavity solver

*HighReCavity — technical note (Sept 2026).  Hardware: Apple M2 Max, Julia 1.12.6, OpenBLAS (8 threads), Float64 throughout.*

## 1. What the reference solver actually solves

The reference (`reference/cavity-flow-spectral-ab2-cn.jl`, kept verbatim; its function bodies are
extracted unchanged into `src/reference.jl`) works with the variable `q` on an `(N+1)×(N+1)`
mapped Chebyshev–Gauss–Lobatto grid (`α = 0.96` arcsin mapping) and the representation

    ψ = (1−x²)(1−y²) q ,      Ψ = Wx Q Wy .

Its operators are the *exact* derivatives of that product:

    laplacian(q)   = D²x_q Q Wy + Wx Q D²y_qᵀ            ( D²x_q = Wx D2x − 4X Dx − 2I )
                   = Δψ  for the degree-(N+2) polynomial ψ = (1−x²)(1−y²)q            … (1)

and the flattened system is (with `c = Δt/(2Re)`)

    A = L − c·Lψ·L = (I − c Δ_N) L ,   L = kron(Wy, D²x_q) + kron(D²y_q, Wx),   Δ_N = kron(I, D2x) + kron(D2y, I),

with every boundary row of `A` replaced by the identity and the boundary RHS set to the lid data
`g`: `q = 0` on left/right/bottom and `q = −½(1−x²)` on the top.  So one time step solves

    interior nodes:  [(I − cΔ_N) ω]_ij = f_ij ,   ω := laplacian(q)           … (2)
    boundary nodes:  q|Γ = g                                                   … (3)

with `f = laplacian(qⁿ) + c·biharmonic(qⁿ) − Δt(3/2 N(qⁿ) − 1/2 N(qⁿ⁻¹))`.

**Three discrete facts** (verified numerically in `test/test_reference_relations.jl`):

* `laplacian(q)` is **not** the plain Chebyshev Laplacian of the nodal streamfunction,
  `D2x Ψ + Ψ D2yᵀ` (ψ has degree N+2 and its nodal values alias onto P_N; the two differ at O(1) for
  generic q, spectrally little for resolved fields).  Consequently the nodal Ψ (an element of P_N
  vanishing on Γ, (N−1)² dof) does not determine q ((N+1)² dof): **the reference state cannot be
  represented by a nodal ψ–ω pair**.
* `biharmonic_matrix = Lψ·L` is exactly `Δ_N ∘ laplacian`, but the *function* `biharmonic(q)`
  used in the RHS equals `Δ_N laplacian(q)` only for `α = 0` (for `α = 0.96` the terms
  `D⁴x_q` and `D2x·D²x_q` are different discrete operators).  The RHS is explicit, so this only
  matters for reproducing it — which we do by calling the same functions.
* Corners: `laplacian(q)` vanishes identically at the four corners (`Wx[1] = Wy[1] = 0`), and
  the corner values of `q` and `ω` never enter any non-corner equation.
* The wall velocity implied by (3), in the polynomial sense of (1), is
  `u(x,1) = ψ_y = −2(1−x²) q(x,1) = (1−x²)²` (regularised lid), zero elsewhere.  This is exactly what
  the reference `velocity(q)` returns on the lid, for *any* interior q.  Evaluated instead with the
  plain `Dy` on nodal Ψ it is **not** `(1−x²)²`: on actual reference trajectories the nodal
  `∂ₙψ − h` is 0.89 (N=32), 0.21 (N=64), 0.50 (N=128) — the Re = 15000 boundary layer is
  under-resolved at these N, so the aliasing is O(1).

## 2. Two influence-matrix formulations

Both use the same Helmholtz stage: (2) is solved on the interior for a *prescribed* boundary
vorticity `ξ = ω|Γ` (corners excluded, `m = 2(Nx−1)+2(Ny−1)`), as a Sylvester equation

    (½I − c D2xᵢᵢ) Ωᵢ + Ωᵢ (½I − c D2yⱼⱼ)ᵀ = fᵢ + c(D2xᵢ,ᵦ Ωᵦ,ⱼ + Ωᵢ,ᵦ D2yⱼ,ᵦᵀ) ,

using the precomputed eigendecomposition of the two 1-D interior blocks (real spectra,
κ(V) ≈ 2 for `D2x_int`).  Nothing of size (N+1)²×(N+1)² is ever formed.  For each unit boundary
vector `e_k` the response is computed once and the closure functional is evaluated, giving the
`m×m` influence matrix `C`; each step solves the particular problem (ξ = 0), then `Cξ = h − d₀`,
then re-solves with `ξ` (Method A; equivalent to adding the correction `ω_c, ψ_c`, same cost) or
adds the stored responses `Rω ξ, Rq ξ` (Method B).

### 2a. `QForm` — exactly the reference system

* "Poisson" stage: recover the interior of `q` from `laplacian(q) = ω` at interior nodes with
  `q|Γ = g`.  Scaling (1) by `Wxᵢᵢ⁻¹` on the left and `Wyⱼⱼ⁻¹` on the right gives the Sylvester
  equation `(Wx⁻¹D²x_q)ᵢᵢ Qᵢ + Qᵢ (Wy⁻¹D²y_q)ⱼⱼᵀ = G`.  `Wx⁻¹D²x_q` has a *complex* spectrum
  (spurious high modes; κ(V) = 3e2 … 3e3 for N = 32 … 192), so the default backend is a real Schur
  (Bartels–Stewart, LAPACK `trsyl`) factorization; a complex-eigendecomposition backend
  (`poisson_mode = :ceigen`) is 4–6× faster and validated to the same accuracy (Sect. 4).
* Closure: **wall-vorticity consistency** `laplacian(q)|Γ = ξ`, i.e. `(C − I)ξ = −d₀` in the
  task's notation (the stored matrix is `C_stored = L_Γ q⁽ᵏ⁾ − e_k`, target `h = 0`).
  The no-slip condition `∂ₙψ = h` is imposed *exactly* through (3), because in the polynomial
  sense `∂ₙψ = −2 w q|Γ` — the normal derivative is carried by the boundary values of `q`.
* Equivalence proof: with unknowns `(q_int, ξ)`, (2)+(3) ⇔ { `ω = Helmholtz(f, ξ)`, `laplacian(q) = ω`
  at interior nodes, `laplacian(q)|Γ = ξ`, `q|Γ = g` } — corners drop out on both sides because
  `laplacian(q)` is zero there and corner values never appear.  Eliminating `q_int` with the
  Sylvester solve leaves the `m×m` Schur complement `C − I`, which is nonsingular iff `A` is.
  Numerically `C = −I + O(c)` (|C+I| = 1.3e-6 at c = 1.7e-8), κ(C) = 1.0 … 2.0 for N ≤ 256, and it
  stays nonsingular for c up to 2.5 (Re = 0.01, Δt = 0.05 tested).

### 2b. `PsiOmegaForm` — the textbook nodal ψ–ω method

* Poisson stage: `D2x ψ + ψ D2yᵀ = ω`, `ψ|Γ = 0` (P_N streamfunction).
* Closure: plain-Chebyshev outward normal derivative `∂ₙψ|Γ = h`, with `h = (1−x²)²` on the lid.

This is a **different discretization** of the same PDE (function space P_N with ψ|Γ = 0 instead of
`(1−x²)(1−y²)P_N`), and it is *not* equivalent to the reference.  Both are spectrally consistent;
for the under-resolved Re = 15000 runs the difference is O(1) (Sect. 4).

## 3. Corners and rank

Corner degrees of freedom are excluded by construction (`BoundaryLayout`): corner ω is decoupled
from every interior stencil, ψ|Γ = 0 makes ∂ₙψ vanish at the corners identically, and in the
q-form the corner values never appear.  Including them would add four zero rows/columns.

* `QForm`: full rank for every N and c tested (`C = −I + O(c)`); LU factorization.
* `PsiOmegaForm`: **rank exactly m − 4 for every c**, not just as c → 0.  The interior Helmholtz
  equation sees the boundary data only through the four rank-1 stencils
  `D2x[ii,1]⊗E[1,jj]`, `D2x[ii,N+1]⊗E[N+1,jj]`, `E[ii,1]⊗D2y[jj,1]ᵀ`, `E[ii,N+1]⊗D2y[jj,N+1]ᵀ`,
  so the four boundary-vorticity distributions
  `ξ = ( D2y[jj,b_y] on wall x=b_x ,  −D2x[ii,b_x] on wall y=b_y )`, one per corner,
  produce an identically zero interior forcing (`|Cξ| ≈ 1e-16·σ_max`, `test_influence.jl`).  They
  are invisible to the interior ω and ψ, and dually four combinations of the ∂ₙψ constraints —
  corner-localised antisymmetric combinations of the two walls meeting at each corner — cannot be
  satisfied.  Singular values show a clean gap (σ_{m−4}/σ₁ > 1e-3, σ_{m−3}/σ₁ < 1e-14).
  Two justified reductions are implemented and reported explicitly (no hidden ε):
  `reduction = :svd` (minimum-norm least squares with the reported rank; the residual of
  `∂ₙψ = h` lies entirely in the 4-dim left null space, e.g. 2.5e-4 for |h| = 1 at N = 16, and it
  shrinks as N grows) and `reduction = :drop4` (drop the four wall-end unknowns/constraints
  (1,2), (1,N), (N+1,2), (N+1,N); the reduced system has κ ≈ 3, all remaining constraints are
  satisfied exactly).  The interior solution is the same for any representative in `ξ + null(C)`.

## 4. Validation against `A⁻¹·rhs` (`scripts/compare_reference.jl`, `notes/comparison_results.txt`)

Same physical state (50 reference steps at Re = 15000, Δt = 5e-4), relative errors in the Frobenius
norm.  The reference's own roundoff floor is the difference between `inv(A)*rhs` and `lu(A)\rhs`.

| N | floor `inv` vs `LU` | q (Method A) | q (Method B) | q (A, ceigen) | ω vs `laplacian(q_ref)` | ψ | u | v | N(q) |
|---|---|---|---|---|---|---|---|---|---|
| 32  | 3.1e-14 | 2.2e-14 | 2.1e-14 | 2.2e-14 | 2.4e-14 | 7.4e-14 | 3.0e-14 | 1.7e-13 | 5.5e-14 |
| 64  | 1.5e-13 | 1.2e-13 | 1.2e-13 | 9.8e-14 | 2.1e-13 | 5.1e-13 | 3.1e-13 | 1.5e-12 | 3.6e-13 |
| 128 | 8.4e-13 | 7.8e-13 | 7.9e-13 | 8.7e-13 | 4.7e-12 | 3.5e-12 | 2.4e-12 | 1.4e-11 | 1.1e-11 |

In every case `q|Γ = g` and nodal `ψ|Γ = 0` hold exactly (0.0), `∂ₙψ = h` holds exactly in the
polynomial sense, `ω = laplacian(q)` holds everywhere to 1e-15 relative, and the interior Helmholtz
residual is at roundoff.  At N = 8 the entire affine map `rhs ↦ q` agrees with `A⁻¹` column by
column to 5.6e-16 (`scripts/diagnostics_small_N.jl`).  After 200 further steps the trajectories
agree to 3e-12 (N=32), 2e-11 (N=64), 8e-11 (N=128) relative — pure roundoff accumulation, consistent
with κ(A) ≈ κ(Δ_N) ~ 1e4–1e6.

`PsiOmegaForm` from the same RHS: ψ differs from the reference by 127 % (N=32), 11 % (N=64),
8.6 % (N=128) relative, interior ω by 20–36 %, wall ω by factors 24–600 (the nodal formulation needs
enormous wall vorticity to move `∂ₙψ` by O(1) through a diffusion length √c ≈ 1e-4 in one step,
whereas the q-form carries `∂ₙψ` in the extra polynomial degrees).  These differences are the
discretization difference of Sect. 1, not an error — but they mean the nodal method must not be
substituted for the reference without a resolution study.

## 5. Performance (`bench/benchmark.jl`, `bench/scaling.jl`; `notes/benchmark_results.txt`, `notes/scaling_results.txt`)

Per-step **linear solve** (median / min, ms; all influence variants allocation-free):

| N | `Ainv*rhs` | LU solves | influence A (Schur) | influence A (ceigen) | influence B (Schur) | RHS |
|---|---|---|---|---|---|---|
| 32  | 0.26 | 0.24 | 0.13 | 0.10 | 0.11 | 0.06 |
| 64  | 2.9  | 5.4  | 0.75 | 0.37 | 0.65 | 0.30 |
| 128 | 29.8 | 84   | 4.6  | 1.19 | 4.3  | 0.59 |
| 160 | 64 (GEMV, same-size random) | – | 8.6 | 2.1 | 7.4 | 0.97 |
| 192 | 121 (GEMV, same-size random) | – | 15.0 | 3.4 | 12.7 | 1.24 |
| 256 | (35 GB, not storable) | – | 36 | 6.9 | 29 | 2.9 |

Initialization and memory:

| N | `inv(A)` init | `Ainv` memory | influence init | influence memory (A / B) |
|---|---|---|---|---|
| 64  | 1.7–2.4 s | 0.13 GB | 0.12 s | 2 MB / 18 MB |
| 128 | 71–100 s  | 2.06 GB | 1.3 s  | 8 MB / 137 MB |
| 192 | ~N⁶ (not run) | 10.3 GB | 6.3 s | 18 MB / 453 MB |
| 256 | – | 32.5 GB | 19 s | 33 MB / 1.06 GB |

Observations:

* The dense GEMV is memory-bandwidth bound at 8·(N+1)⁴ bytes per step; the influence solve is
  ~10 GEMMs of size (N−1)³ plus an `m×m` LU solve (0.05 ms at N=128).  The crossover is already at
  N = 32; at N = 128 Method A (Schur) is 6.5× and Method A (ceigen) 25× faster than `Ainv*rhs`, and
  initialization is 60× cheaper.  Beyond N ≈ 200 the dense inverse is not an option at all.
* With the Schur backend, `trsyl` (unblocked LAPACK) is ~80 % of the q-Poisson cost (1.85 of 2.06 ms
  at N=128) and dominates Method A; the eigen-based Helmholtz solve (4 GEMMs) costs only 0.19 ms.
  The complex-eigen backend removes this (0.36 ms) at an accuracy loss bounded by κ(V)·ε ≈ 1e-13
  … 1e-12 — below the reference's own floor in practice (table above).
* Method B is **not** faster than Method A + ceigen: the two response GEMVs stream 2·(N+1)²·m·8 B
  (129 MB at N=128, 453 MB at N=192) per step and are memory bound.  It helps only against the
  Schur backend (one `trsyl` saved).  It is kept as an option (and would be the natural GPU
  kernel), but on this CPU Method A with `:ceigen` is the recommended configuration.
* GPU: the M2 Max's Metal backend has no Float64, so per the "Float64 first" requirement no GPU
  path was built; all timings are CPU (no host–device transfers).  The per-step work is
  ~10 GEMMs of (N−1)³ plus small GEMVs, i.e. entirely batched-BLAS-friendly if a Float64 GPU is
  available.
* Full step cost at N=128: reference 32 ms (allocating RHS + GEMV) vs 5.7 ms (Method A, Schur) /
  3.8 ms median, 1.9 ms min (Method A, ceigen; the spread is BLAS-thread scheduling on the many
  small GEMMs); the RHS itself is 0.59 ms.

## 6. Code map

* `src/reference.jl` — verbatim reference functions, `ReferenceSystem`, `step_reference!`.
* `src/psiomega.jl` — ψ–ω routines (velocities, vorticity derivatives, convection, Laplacians).
* `src/sylvester.jl` — `SylvesterSolver` (`:eigen`, `:schur`, `:ceigen`), `SeparableHelmholtzSolver`,
  `SeparablePoissonSolver` (P_N), `SeparableQPoissonSolver` (reference-exact).
* `src/boundary.jl` — `BoundaryLayout`, gather/scatter, normal derivatives (nodal and polynomial), lid data.
* `src/influence.jl` — `InfluenceSolver{QForm|PsiOmegaForm}`, matrix build, SVD diagnostics,
  factorization/reductions, `solve_timestep_linear!` (Methods A/B).
* `src/timestep.jl` — allocation-free reference RHS, `QState`/`step_influence!`, `PsiOmegaState`/`step_psiomega!`.
* `test/` — 401 tests (boundary maps, normal derivatives, Poisson/Helmholtz/q-Poisson vs dense,
  influence matrix rank structure, full linear solve vs `A⁻¹`, 20-step trajectories at N = 32, 64).
* `scripts/compare_reference.jl`, `scripts/diagnostics_small_N.jl`, `scripts/run_cavity.jl`,
  `bench/benchmark.jl`, `bench/scaling.jl`.
