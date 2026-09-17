# Discrete operators of the q formulation.
#
# The streamfunction is represented as  ψ = (1−x²)(1−y²) q,  Ψ = W Q W  with W = diag(1−x²).
# All x-derivatives act from the left (D·Q), all y-derivatives from the right (Q·Dᵀ);
# Q[i, j] = q(x_i, y_j).  Derivatives of ψ are taken by the product rule *exactly* (this is
# what makes the formulation different from a nodal ψ–ω method and must be preserved):
#
#   ∂x[(1−x²)q]   = (1−x²) q'   − 2x q             → Dq  = W D1 − 2X
#   ∂xx[(1−x²)q]  = (1−x²) q''  − 4x q'  − 2q      → D2q = W D2 − 4X D1 − 2I
#   ∂⁴x[(1−x²)q]  = (1−x²) q'''' − 8x q''' − 12q''  → D4q = W D2² − 8X D2 D1 − 12 D2

struct CavityOperators{T<:AbstractFloat}
    grid::ChebyshevGrid{T}
    D1::Matrix{T}         # plain Chebyshev d/dx
    D2::Matrix{T}         # plain Chebyshev d²/dx²  (the Laplacian acting on ω is D2·Ω + Ω·D2ᵀ)
    Dq::Matrix{T}         # product-rule operators acting on q (see above)
    D2q::Matrix{T}
    D4q::Matrix{T}
    w::Vector{T}          # 1 − x²  (zero at the walls)
end

function CavityOperators(grid::ChebyshevGrid{T}) where {T}
    N = grid.N; x = grid.x
    D1, D2 = diff_matrices(grid)
    W = Diagonal(1 .- x.^2); X = Diagonal(x)
    Dq  = W * D1 - 2 * X
    D2q = W * D2 - 4 * X * D1 - 2 * I(N+1)
    D4q = W * D2^2 - 8 * X * D2 * D1 - 12 * D2
    return CavityOperators{T}(grid, D1, D2, Dq, D2q, D4q, W.diag)
end

# ---------------------------------------------------------------------------------------
# In-place field operators (tmp arrays are (N+1)×(N+1) scratch).
# ---------------------------------------------------------------------------------------

"""
    laplacian!(ω, q, ops, tmp)  —  ω = Δψ = D2q·Q·W + W·Q·D2qᵀ
"""
function laplacian!(ω, q, ops::CavityOperators, tmp)
    w = ops.w
    mul!(tmp, q, transpose(ops.D2q));  ω .= w .* tmp          # (1−x²) ∂yy[(1−y²)q]
    mul!(tmp, ops.D2q, q);             ω .+= tmp .* w'        # ∂xx[(1−x²)q] (1−y²)
    return ω
end

"""
    biharmonic!(B, q, ops, tmp1, tmp2)  —  B = ψ_xxxx + ψ_yyyy + 2ψ_xxyy
                                             = D4q·Q·W + W·Q·D4qᵀ + D2·(W·Q·D2qᵀ) + (D2q·Q·W)·D2ᵀ
"""
function biharmonic!(B, q, ops::CavityOperators, tmp1, tmp2)
    w = ops.w
    mul!(tmp1, q, transpose(ops.D4q));  B .= w .* tmp1
    mul!(tmp1, ops.D4q, q);             B .+= tmp1 .* w'
    mul!(tmp1, q, transpose(ops.D2q));  tmp2 .= w .* tmp1;   mul!(tmp1, ops.D2, tmp2);            B .+= tmp1
    mul!(tmp1, ops.D2q, q);             tmp2 .= tmp1 .* w';  mul!(tmp1, tmp2, transpose(ops.D2)); B .+= tmp1
    return B
end

"""
    convection!(C, q, ops, W)  —  C = u ω_x + v ω_y   with  u = ψ_y = W·Q·Dqᵀ,  v = −ψ_x = −Dq·Q·W,
                                   ω = Δψ (laplacian!),  ω_x = D1·Ω,  ω_y = Ω·D1ᵀ
"""
function convection!(C, q, ops::CavityOperators, W)
    w = ops.w
    mul!(W.A, q, transpose(ops.Dq));  W.u .= w .* W.A
    mul!(W.A, ops.Dq, q);             W.v .= .-(W.A .* w')
    laplacian!(W.B, q, ops, W.A)                          # ω
    mul!(W.A, ops.D1, W.B)                                # ω_x
    mul!(W.C, W.B, transpose(ops.D1))                     # ω_y
    C .= W.u .* W.A .+ W.v .* W.C
    return C
end

# ---------------------------------------------------------------------------------------
# Allocating post-processing versions (same operators)
# ---------------------------------------------------------------------------------------
streamfunction(q::AbstractMatrix, ops::CavityOperators) = ops.w .* q .* ops.w'
vorticity(q::AbstractMatrix, ops::CavityOperators) = laplacian!(similar(q), q, ops, similar(q))
function velocity(q::AbstractMatrix, ops::CavityOperators)
    u = ops.w .* (q * transpose(ops.Dq))          # u =  ψ_y
    v = .-((ops.Dq * q) .* ops.w')                # v = −ψ_x
    return u, v
end

"""
    lid_boundary_values(ops) -> g  (vector in BoundaryLayout order)

Dirichlet data for q: zero on the left/right/bottom walls, −½(1−x²) on the lid.  In the
polynomial sense ψ_y|_{y=1} = −2(1−x²) q(x,1) = (1−x²)², i.e. the regularised lid velocity
u(x, 1) = (1−x²)²; the corner values are irrelevant (see BoundaryLayout).
"""
function lid_boundary_values(ops::CavityOperators{T}, L) where {T}
    g = zeros(T, L.m)
    x = ops.grid.x
    for (k, i) in enumerate(2:L.N)
        g[L.top[k]] = -1/2 * (1 - x[i]^2)
    end
    return g
end
