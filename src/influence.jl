# ============================================================================
# Phase 6/7 — influence-matrix solver.
#
# Linear problem solved each time step (c = Δt/(2Re)):
#
#     (I − cΔ) ω = f     (interior),   ω|Γ = ξ  (unknown, m values, corners excluded)
#     "Poisson"  stage   ψ (or q) from ω with prescribed boundary data g
#     closure    stage   d(ψ, ω) = h   (m equations)
#
# Two formulations share this machinery (type parameter `F`):
#
#   QForm       — reproduces the reference dense system EXACTLY (to roundoff).
#                 Poisson stage: interior of q from  laplacian(q) = ω, q|Γ = g (lid data).
#                 Closure:       laplacian(q)|Γ − ω|Γ = 0   (wall-vorticity consistency).
#                 (The normal-derivative condition ∂ₙψ = h is imposed *exactly* through
#                  q|Γ = g, since ∂ₙψ = −2 w q|Γ in the polynomial sense.)
#
#   PsiOmegaForm — textbook P_N streamfunction–vorticity influence matrix.
#                 Poisson stage: Dx²ψ + ψDy²ᵀ = ω, ψ|Γ = 0.
#                 Closure:       ∂ₙψ|Γ = h  (plain Chebyshev normal derivative).
#                 This is a DIFFERENT discretization from the reference (see the note).
#
# In both cases the closure is affine in ξ:  d(ξ) = d₀ + C ξ,  and the influence
# matrix C is built once, column by column, from unit boundary-vorticity responses.
# ============================================================================

abstract type Formulation end
struct QForm <: Formulation end
struct PsiOmegaForm <: Formulation end

mutable struct InfluenceSolver{F<:Formulation, T<:AbstractFloat, P}
    grid::ChebyshevGrid{T,2}
    ops::CavityOperators{T,Matrix{T}}
    Δt::T
    Re::T
    c::T
    layout::BoundaryLayout
    helm::SeparableHelmholtzSolver{T}
    pois::P
    g::Vector{T}                 # Dirichlet data for the Poisson stage (boundary of q or ψ)
    h::Vector{T}                 # closure target
    C::Matrix{T}                 # influence matrix (m×m)
    Cfact::Any                   # LU (full rank) or SVD pseudo-inverse data
    svals::Vector{T}             # singular values of C (diagnostic)
    rank::Int
    # Method B response matrices ((Nx+1)(Ny+1) × m); empty unless built
    Rω::Matrix{T}
    RX::Matrix{T}
    # work
    ξ::Vector{T}
    d0::Vector{T}
    ωwork::Matrix{T}
    Xwork::Matrix{T}
    ω0::Matrix{T}
    X0::Matrix{T}
end

_poisson_stage(::Type{QForm}, ops; mode) = SeparableQPoissonSolver(ops; mode)
_poisson_stage(::Type{PsiOmegaForm}, ops; mode) = SeparablePoissonSolver(ops; mode)

# Poisson stage: boundary of X holds g; interior overwritten.
_solve_poisson_stage!(X, ω, S::InfluenceSolver{QForm}) = solve_qpoisson_dirichlet!(X, ω, S.pois)
_solve_poisson_stage!(X, ω, S::InfluenceSolver{PsiOmegaForm}) = solve_poisson_dirichlet!(X, ω, S.pois)

# Closure functional d(X, ω) (affine in the boundary vorticity)
function closure!(d, X, ω, S::InfluenceSolver{QForm})
    laplacian_boundary!(d, X, S.ops, S.layout)         # (L q)|Γ
    @inbounds for k in 1:S.layout.m
        d[k] -= ω[S.layout.points[k]]                    # − ω|Γ
    end
    return d
end
closure!(d, X, ω, S::InfluenceSolver{PsiOmegaForm}) = normal_derivative_boundary!(d, X, S.ops, S.layout)

"""
    InfluenceSolver(F, grid, ops, Δt, Re; helmholtz_mode=:eigen, poisson_mode, build_response=false, verbose=true)

Builds the Helmholtz and Poisson-stage decompositions, the boundary layout, the
influence matrix `C` (m×m, m = 2(Nx-1)+2(Ny-1)) and its factorization.  Prints
rank/conditioning diagnostics when `verbose`.
"""
function InfluenceSolver(::Type{F}, grid::ChebyshevGrid{T,2}, ops::CavityOperators{T,Matrix{T}}, Δt::T, Re::T;
                         helmholtz_mode::Symbol = :eigen,
                         poisson_mode::Symbol = (F === QForm ? :schur : :eigen),
                         build_response::Bool = false, verbose::Bool = true,
                         rank_tol = nothing, reduction::Symbol = :svd) where {F<:Formulation, T<:AbstractFloat}
    Nx, Ny = grid.Ns
    c = T(0.5) * Δt / Re
    layout = BoundaryLayout(Nx, Ny)
    helm = SeparableHelmholtzSolver(ops, c; mode = helmholtz_mode)
    pois = _poisson_stage(F, ops; mode = poisson_mode)
    if F === QForm
        g = lid_q_boundary(grid, layout)
        h = zeros(T, layout.m)
    else
        g = zeros(T, layout.m)
        h = lid_normal_derivative_target(grid, ops, layout)
    end
    m = layout.m
    S = InfluenceSolver{F,T,typeof(pois)}(grid, ops, Δt, Re, c, layout, helm, pois, g, h,
            zeros(T, m, m), nothing, T[], 0, zeros(T, 0, 0), zeros(T, 0, 0),
            zeros(T, m), zeros(T, m),
            zeros(T, Nx+1, Ny+1), zeros(T, Nx+1, Ny+1), zeros(T, Nx+1, Ny+1), zeros(T, Nx+1, Ny+1))
    build_influence_matrix!(S; build_response)
    factorize_influence!(S; rank_tol, reduction, verbose)
    return S
end

InfluenceSolver(grid, ops, Δt, Re; kw...) = InfluenceSolver(QForm, grid, ops, Δt, Re; kw...)

"""
    build_influence_matrix!(S; build_response=false)

For each boundary unit vector e_k: ω⁽ᵏ⁾ = Helmholtz(f=0, ω|Γ=e_k), X⁽ᵏ⁾ = Poisson(ω⁽ᵏ⁾, X|Γ=0),
C[:, k] = closure(X⁽ᵏ⁾, ω⁽ᵏ⁾).  Optionally stores the flattened responses (Method B).
"""
function build_influence_matrix!(S::InfluenceSolver{F,T}; build_response::Bool = false) where {F,T}
    m = S.layout.m
    n = length(S.ωwork)
    ω = S.ωwork; X = S.Xwork
    zerof = S.ω0            # f = 0 (only interior of f is read)
    fill!(zerof, zero(T))
    if build_response
        S.Rω = zeros(T, n, m); S.RX = zeros(T, n, m)
    end
    ek = zeros(T, m)
    d = zeros(T, m)
    for k in 1:m
        fill!(ek, zero(T)); ek[k] = one(T)
        fill!(ω, zero(T)); scatter!(ω, ek, S.layout)
        solve_helmholtz_dirichlet!(ω, zerof, S.helm)
        fill!(X, zero(T))                                   # X|Γ = 0 (homogeneous response)
        _solve_poisson_stage!(X, ω, S)
        closure!(d, X, ω, S)
        S.C[:, k] .= d
        if build_response
            S.Rω[:, k] .= vec(ω); S.RX[:, k] .= vec(X)
        end
    end
    return S
end

"""
    factorize_influence!(S; rank_tol=nothing, reduction=:svd, verbose=true)

SVD diagnostic of `C` (rank, conditioning, smallest singular values).

* full rank            → LU factorization.
* rank deficient       → `reduction = :svd`  : minimum-norm least-squares solve with an
                          *explicit* truncated SVD of the reported rank (the null-space
                          directions of ω|Γ are invisible to the interior equations, so
                          the interior ω and ψ do not depend on this choice);
                         `reduction = :drop4` : drop the four wall-end unknowns and
                          constraints (1,2), (1,Ny), (Nx+1,2), (Nx+1,Ny) and LU-factorize
                          the remaining square system (classical corner treatment).
The rank tolerance is `m·eps·σ_max` unless `rank_tol` is given; it is printed.
"""
function factorize_influence!(S::InfluenceSolver{F,T}; rank_tol = nothing, reduction::Symbol = :svd,
                              verbose::Bool = true) where {F,T}
    C = S.C
    m = size(C, 1)
    sv = svdvals(C)
    S.svals = sv
    tol = rank_tol === nothing ? m * eps(T) * sv[1] : rank_tol
    r = count(>(tol), sv)
    S.rank = r
    if verbose
        @printf("Influence matrix (%s): %d×%d, numerical rank %d (tol %.2e), σ_max = %.3e, σ_min = %.3e, κ = %.3e\n",
                string(F), m, m, r, tol, sv[1], sv[end], sv[1]/sv[end])
        @printf("  smallest singular values: %s\n", join([@sprintf("%.3e", s) for s in sv[max(1,end-5):end]], ", "))
    end
    if r == m
        S.Cfact = lu(C)
    elseif reduction == :svd
        verbose && @warn "Influence matrix is rank deficient ($r < $m); using truncated-SVD minimum-norm least squares"
        U, s, V = svd(C)
        S.Cfact = (kind = :svd, U = U[:, 1:r], s = s[1:r], V = V[:, 1:r], Unull = U[:, r+1:m], Vnull = V[:, r+1:m],
                   tmp = zeros(T, r))
    elseif reduction == :drop4
        L = S.layout
        drop = [L.left[1], L.left[end], L.right[1], L.right[end]]
        keep = setdiff(1:m, drop)
        Cr = C[keep, keep]
        svr = svdvals(Cr)
        verbose && @printf("  reduced system (drop 4 wall-end dofs): %d×%d, κ = %.3e\n", length(keep), length(keep), svr[1]/svr[end])
        S.Cfact = (kind = :drop4, keep = keep, lu = lu(Cr), tmp = zeros(T, length(keep)))
    else
        error("unknown reduction $reduction")
    end
    return S
end

function _solve_influence!(ξ, S::InfluenceSolver, rhs)
    f = S.Cfact
    if f isa LU
        ξ .= rhs
        ldiv!(f, ξ)
    elseif f.kind == :svd
        mul!(f.tmp, transpose(f.U), rhs)
        f.tmp ./= f.s
        mul!(ξ, f.V, f.tmp)
    else # :drop4
        f.tmp .= view(rhs, f.keep)
        ldiv!(f.lu, f.tmp)
        fill!(ξ, zero(eltype(ξ)))
        view(ξ, f.keep) .= f.tmp
    end
    return ξ
end

"""
    solve_timestep_linear!(ω, X, f, S::InfluenceSolver; method = :resolve)

Given the explicit right-hand side `f` (interior values used), computes the new
vorticity `ω` (full grid; corners zero) and the new `X` (= q for `QForm`, = ψ for
`PsiOmegaForm`; boundary = `S.g`).

`method = :resolve`  — Method A: particular solve with ω|Γ=0 → d₀; solve Cξ = h − d₀;
                        then one more Helmholtz+Poisson pair with ω|Γ = ξ.
`method = :response` — Method B: ω = ω₀ + Rω ξ, X = X₀ + RX ξ (requires `build_response=true`).
"""
function solve_timestep_linear!(ω::AbstractMatrix{T}, X::AbstractMatrix{T}, f::AbstractMatrix{T},
                                S::InfluenceSolver{F,T}; method::Symbol = :resolve) where {F,T}
    L = S.layout
    ξ = S.ξ; d0 = S.d0
    # --- particular solution: ω|Γ = 0 ---
    ω0 = (method == :response) ? S.ω0 : ω
    X0 = (method == :response) ? S.X0 : X
    fill!(ξ, zero(T)); scatter!(ω0, ξ, L)
    solve_helmholtz_dirichlet!(ω0, f, S.helm)
    scatter!(X0, S.g, L)
    _solve_poisson_stage!(X0, ω0, S)
    closure!(d0, X0, ω0, S)
    # --- boundary vorticity: C ξ = h − d₀ ---
    d0 .= S.h .- d0
    _solve_influence!(ξ, S, d0)
    # --- correction ---
    if method == :resolve
        scatter!(ω, ξ, L)
        solve_helmholtz_dirichlet!(ω, f, S.helm)
        scatter!(X, S.g, L)
        _solve_poisson_stage!(X, ω, S)
    elseif method == :response
        isempty(S.Rω) && error("response matrices not built; construct with build_response=true")
        mul!(vec(ω), S.Rω, ξ); ω .+= ω0
        mul!(vec(X), S.RX, ξ); X .+= X0
    else
        error("unknown method $method")
    end
    return ω, X
end
