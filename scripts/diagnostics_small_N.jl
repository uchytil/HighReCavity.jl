# Exact algebra check at small N: assemble the complete affine map rhs ↦ q of both
# solvers column by column and compare, and exhibit the Schur-complement identity
# behind the influence matrix.   julia --project=. scripts/diagnostics_small_N.jl
using HighReCavity, LinearAlgebra, Printf, Logging

N = 8; Δt = 0.0005; Re = 15000.0
grid = ChebyshevGrid((N, N), (0.96, 0.96), Float64)
ops = CavityOperators(Matrix{Float64}, grid)
sys = ReferenceSystem(grid, ops, Δt, Re)
S = InfluenceSolver(QForm, grid, ops, Δt, Re)
L = S.layout
n = (N+1)^2
li = LinearIndices((N+1, N+1)); int = vec(li[2:N, 2:N]); bnd = setdiff(1:n, int)

# --- 1. full affine map: q = F_int * rhs_int + q_g  (rhs boundary rows are overwritten by the lid data) ---
gflat = zeros(n); apply_bcs_rhs!(gflat, sys.walls_idxs_flat, sys.top_idxs_flat, grid)
F_ref = sys.Ainv[:, int]
qg_ref = sys.Ainv * gflat
F_inf = zeros(n, length(int)); q = zeros(N+1, N+1); ω = zeros(N+1, N+1)
solve_timestep_linear!(ω, q, zeros(N+1, N+1), S); qg_inf = vec(copy(q))
for (k, idx) in enumerate(int)
    e = zeros(N+1, N+1); e[idx] = 1
    solve_timestep_linear!(ω, q, e, S)
    F_inf[:, k] .= vec(q) .- qg_inf
end
@printf("full map    |F_inf − A⁻¹[:,int]|_max = %.2e   (|A⁻¹| max %.2e)\n", maximum(abs.(F_inf - F_ref)), maximum(abs.(F_ref)))
@printf("affine part |q_g,inf − A⁻¹ g|_max     = %.2e\n", maximum(abs.(qg_inf - qg_ref)))

# --- 2. Schur-complement identity.  Unknowns (q_int, ξ = ω|Γ), equations:
#   (a) interior Helmholtz rows of A:  (I − cΔ_N)(L q) = rhs  ⇔  L q = ω(ξ, rhs)   [ω from the Helmholtz solve]
#   (b) closure                       (L q)|Γ = ξ
# Eliminating q_int with (a) gives  (C − I) ξ = −d₀ ... which is exactly what the solver does.  Verify that the
# influence matrix stored in S equals the boundary rows of L applied to the responses:
Lm = laplacian_matrix(ops)
LΓ = Lm[[li[p] for p in L.points], :]
# response q⁽ᵏ⁾ for unit boundary vorticity e_k (ξ = e_k, rhs = 0, q|Γ = 0)
Cchk = zeros(L.m, L.m)
for k in 1:L.m
    ξ = zeros(L.m); ξ[k] = 1
    fill!(ω, 0); scatter!(ω, ξ, L); solve_helmholtz_dirichlet!(ω, zeros(N+1, N+1), S.helm)
    fill!(q, 0); solve_qpoisson_dirichlet!(q, ω, S.pois)
    Cchk[:, k] .= LΓ * vec(q) .- ξ
end
@printf("influence   |C_stored − (L_Γ q⁽ᵏ⁾ − e_k)|_max = %.2e\n", maximum(abs.(Cchk - S.C)))
@printf("influence   C = −I + O(c):  |C + I|_max = %.2e  (c = %.2e)\n", maximum(abs.(S.C + I)), S.c)
sv = svdvals(S.C)
@printf("influence   σ(C) ∈ [%.6f, %.6f]\n", sv[end], sv[1])
