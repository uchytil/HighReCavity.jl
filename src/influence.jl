# Influence-matrix solve of one implicit step (Method A: fresh re-solve after the boundary
# correction).  With c = dt/(2 Re_internal) and ω := Δψ = L q, the implicit system is
#
#     (I − cΔ) ω = f      at interior nodes                          (Helmholtz stage)
#     L q = ω              at interior nodes,   q|Γ = g (lid data)   (q-Poisson stage)
#     (L q)|Γ = ω|Γ        wall-vorticity consistency                (closure)
#
# The unknown wall vorticity ξ = ω|Γ (m = 4(N−1) non-corner wall values) enters linearly:
# (L q)|Γ = d₀ + Cξ, so the closure is  (C − I) ξ = −d₀.  The m×m matrix  M = C − I  is built
# once from unit responses and LU-factorized; it equals −I + O(c) and is nonsingular exactly
# when the original dense system is.  The no-slip condition is imposed through q|Γ = g,
# because in this representation ∂ₙψ = −2(1−x²) q|Γ exactly.

"""
Ordered non-corner wall nodes: left (1, 2:N), right (N+1, 2:N), bottom (2:N, 1), top (2:N, N+1).
Corners are excluded: no interior stencil touches them, L q vanishes there, and q's corner
values never enter any equation.
"""
struct BoundaryLayout
    N::Int
    m::Int
    points::Vector{CartesianIndex{2}}
    left::UnitRange{Int}; right::UnitRange{Int}; bottom::UnitRange{Int}; top::UnitRange{Int}
    corners::Vector{CartesianIndex{2}}
end

function BoundaryLayout(N::Int)
    n = N - 1
    left, right, bottom, top = 1:n, n+1:2n, 2n+1:3n, 3n+1:4n
    pts = Vector{CartesianIndex{2}}(undef, 4n)
    for (k, j) in enumerate(2:N)
        pts[left[k]] = CartesianIndex(1, j);   pts[right[k]] = CartesianIndex(N+1, j)
        pts[bottom[k]] = CartesianIndex(j, 1); pts[top[k]] = CartesianIndex(j, N+1)
    end
    corners = [CartesianIndex(1, 1), CartesianIndex(N+1, 1), CartesianIndex(1, N+1), CartesianIndex(N+1, N+1)]
    return BoundaryLayout(N, 4n, pts, left, right, bottom, top, corners)
end

"Write wall vector v onto the four edges of M (corners set to zero); interior untouched."
function set_walls!(M::AbstractMatrix, v::AbstractVector, L::BoundaryLayout)
    @inbounds for k in 1:L.m
        M[L.points[k]] = v[k]
    end
    @inbounds for c in L.corners
        M[c] = zero(eltype(M))
    end
    return M
end

"""
    wall_laplacian!(d, q, ops, L)  —  d = (L q)|Γ,  the wall rows/columns of  D2q·Q·W + W·Q·D2qᵀ

O(N²): at a wall only the derivative *normal* to it survives because W = 0 there (the
W-weighted terms are kept so that this is exactly the wall part of `laplacian!`).
"""
function wall_laplacian!(d::AbstractVector{T}, q::AbstractMatrix{T}, ops::CavityOperators{T}, L::BoundaryLayout) where {T}
    N = L.N; D2q = ops.D2q; w = ops.w
    @inbounds for (k, j) in enumerate(2:N)
        d[L.left[k]]   = w[j] * dot(view(D2q, 1, :), view(q, :, j))   + w[1]   * dot(view(q, 1, :), view(D2q, j, :))
        d[L.right[k]]  = w[j] * dot(view(D2q, N+1, :), view(q, :, j)) + w[N+1] * dot(view(q, N+1, :), view(D2q, j, :))
        d[L.bottom[k]] = w[j] * dot(view(q, j, :), view(D2q, 1, :))   + w[1]   * dot(view(D2q, j, :), view(q, :, 1))
        d[L.top[k]]    = w[j] * dot(view(q, j, :), view(D2q, N+1, :)) + w[N+1] * dot(view(D2q, j, :), view(q, :, N+1))
    end
    return d
end

struct InfluenceSolver{T<:AbstractFloat, H<:HelmholtzSolver{T}, P<:QPoissonSolver{T}}
    ops::CavityOperators{T}
    layout::BoundaryLayout
    helm::H
    pois::P
    g::Vector{T}                    # lid data q|Γ
    M::Matrix{T}                    # influence matrix  C − I
    Mlu::LU{T,Matrix{T},Vector{Int}}
    condM::T                        # condition number of M (diagnostic)
    ξ::Vector{T}; d::Vector{T}      # wall vorticity, closure residual
    zero_f::Matrix{T}               # f = 0 for the unit responses
end

"""
    InfluenceSolver(ops, c; backend)   with  c = dt / (2 Re_internal)

Builds the Helmholtz and q-Poisson decompositions (`backend` selects the latter) and the influence matrix
M[:, k] = (L q⁽ᵏ⁾)|Γ − e_k, where (ω⁽ᵏ⁾, q⁽ᵏ⁾) is the response to f = 0, ω|Γ = e_k, q|Γ = 0.
"""
function InfluenceSolver(ops::CavityOperators{T}, c::T; backend::Symbol = :ceigen) where {T}
    N = ops.grid.N
    L = BoundaryLayout(N)
    helm = HelmholtzSolver(ops, c)
    pois = QPoissonSolver(ops, backend)
    g = lid_boundary_values(ops, L)
    m = L.m
    M = zeros(T, m, m)
    ω = zeros(T, N+1, N+1); q = zeros(T, N+1, N+1); zero_f = zeros(T, N+1, N+1)
    ek = zeros(T, m); d = zeros(T, m)
    for k in 1:m
        ek .= 0; ek[k] = 1
        set_walls!(ω, ek, L);  helmholtz!(ω, zero_f, helm)          # (I − cΔ) ω⁽ᵏ⁾ = 0,  ω⁽ᵏ⁾|Γ = e_k
        fill!(q, 0);           qpoisson!(q, ω, pois)                 # L q⁽ᵏ⁾ = ω⁽ᵏ⁾,      q⁽ᵏ⁾|Γ = 0
        wall_laplacian!(d, q, ops, L)
        M[:, k] .= d .- ek
    end
    sv = svdvals(M)
    condM = sv[1] / sv[end]
    condM > 1e12 && @warn "influence matrix is nearly singular (κ = $condM)"
    return InfluenceSolver{T,typeof(helm),typeof(pois)}(ops, L, helm, pois, g, M, lu(M), condM, zeros(T, m), d, zero_f)
end

"""
    influence_solve!(q, ω, f, S)

Given the explicit right-hand side f (interior values), computes q^{n+1} (interior; walls set
to the lid data) and the corresponding vorticity ω = L q (all nodes, corners zero).
"""
function influence_solve!(q::AbstractMatrix{T}, ω::AbstractMatrix{T}, f::AbstractMatrix{T}, S::InfluenceSolver{T}) where {T}
    L = S.layout; ξ = S.ξ; d = S.d
    # 1. particular solution with zero wall vorticity → closure residual d₀ = (L q₀)|Γ
    ξ .= 0
    set_walls!(ω, ξ, L);    helmholtz!(ω, f, S.helm)
    set_walls!(q, S.g, L);  qpoisson!(q, ω, S.pois)
    wall_laplacian!(d, q, S.ops, L)
    # 2. wall vorticity from the closure  (C − I) ξ = −d₀
    ξ .= .-d
    ldiv!(S.Mlu, ξ)
    # 3. re-solve with the correct wall vorticity (Method A)
    set_walls!(ω, ξ, L);    helmholtz!(ω, f, S.helm)
    set_walls!(q, S.g, L);  qpoisson!(q, ω, S.pois)
    return q, ω
end
