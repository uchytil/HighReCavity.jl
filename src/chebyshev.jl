# Mapped Chebyshev–Gauss–Lobatto grid and differentiation matrices (1-D; the square
# cavity uses the same grid in x and y).  Arithmetic identical to the original solver.

struct ChebyshevGrid{T<:AbstractFloat}
    N::Int
    η::Vector{T}     # Gauss–Lobatto nodes  η_k = −cos(πk/N),  k = 0..N   (−1 … 1)
    x::Vector{T}     # mapped nodes  x = asin(αη)/asin(α)  (α = 0: x = η); clusters less at the walls
    α::T
end

function ChebyshevGrid(N::Int, α::T) where {T<:AbstractFloat}
    η = -cos.(T(π) .* T.(0:N) ./ T(N))
    x = α > 0 ? asin.(α .* η) ./ asin(α) : copy(η)
    return ChebyshevGrid{T}(N, η, x, α)
end

"""
    diff_matrices(grid) -> (D1, D2)

First- and second-derivative matrices in the physical coordinate x.  Standard Chebyshev
matrix in η, chain rule through the mapping (d/dx = (dη/dx) d/dη), D2 = D1².
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

Barycentric interpolation from the grid nodes to arbitrary points in [-1, 1]
(post-processing / plotting only).
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
