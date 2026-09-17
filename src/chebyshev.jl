# Mapped Chebyshev–Gauss–Lobatto grid and differentiation matrices in one dimension; the
# square cavity uses the same grid in x and y.
#
# The Gauss–Lobatto nodes η_k = −cos(πk/N) cluster like O(N⁻²) at the ends of [−1, 1].  The
# Kosloff–Tal-Ezer mapping  x = asin(αη)/asin(α),  0 ≤ α < 1,  spreads the nodes near the walls
# (spacing → O(N⁻¹) as α → 1).  The mapping is analytic only for |αη| < 1, so α close to 1
# trades convergence rate for a milder explicit time-step restriction.  α = 0 gives the
# unmapped Chebyshev grid.

"""
    ChebyshevGrid(N, α)

One-dimensional mapped Chebyshev–Gauss–Lobatto grid with `N+1` nodes: `η` are the
Gauss–Lobatto nodes −cos(πk/N) and `x = asin(αη)/asin(α)` the physical nodes, 0 ≤ `α` < 1
(`α = 0`: unmapped).  The cavity uses the same grid in both directions.
"""
struct ChebyshevGrid{T<:AbstractFloat}
    N::Int
    η::Vector{T}     # Gauss–Lobatto nodes  η_k = −cos(πk/N),  k = 0..N   (−1 … 1)
    x::Vector{T}     # mapped nodes  x = asin(αη)/asin(α)  (α = 0: x = η)
    α::T
end

function ChebyshevGrid(N::Int, α::T) where {T<:AbstractFloat}
    η = -cos.(T(π) .* T.(0:N) ./ T(N))
    x = α > 0 ? asin.(α .* η) ./ asin(α) : copy(η)
    return ChebyshevGrid{T}(N, η, x, α)
end

"""
    diff_matrices(grid) -> (D1, D2)

Collocation derivative matrices in the physical coordinate x, acting on nodal values: the
standard Chebyshev differentiation matrix in η, scaled row-wise by dη/dx (chain rule through
the mapping), and D2 = D1·D1.
"""
function diff_matrices(grid::ChebyshevGrid{T}) where {T}
    N = grid.N; η = grid.η; α = grid.α
    c = [2; ones(N-1); 2] .* T.((-1).^(0:N))
    X = repeat(η, 1, N+1)
    dX = X - X'
    D = (c * (1 ./ c)') ./ (dX .+ I(N+1))
    D = D - diagm(vec(sum(D, dims = 2)))
    if α > 0
        L = asin(α)
        D = (@. (L * sqrt(1 - (α * η)^2)) / α) .* D          # dη/dx = L√(1−α²η²)/α
    end
    return D, D * D
end

"""
    interp_matrix(x_target, grid) -> M

Barycentric interpolation matrix from the grid nodes to arbitrary points `x_target` in
[-1, 1]: values at the targets are `M * values_at_nodes`.  For a 2-D field F on the grid,
`Mx * F * My'` evaluates it on the tensor grid of the targets (e.g. for plotting).
"""
function interp_matrix(x_target::AbstractVector{T}, grid::ChebyshevGrid{T}) where {T}
    N = grid.N; η = grid.η; α = grid.α
    w = ones(T, N+1); w[1] = w[end] = T(0.5); w .*= T.((-1).^(0:N))
    η_target = α > 0 ? sin.(asin(α) .* x_target) ./ α : collect(x_target)
    M = zeros(T, length(x_target), N+1)
    for i in eachindex(x_target)
        dη = η_target[i] .- η
        k = findfirst(iszero, dη)
        if k !== nothing
            M[i, k] = one(T)
        else
            ww = w ./ dη
            M[i, :] = ww ./ sum(ww)
        end
    end
    return M
end
