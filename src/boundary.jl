# ============================================================================
# Phase 4/5 — boundary degrees of freedom and wall-normal derivatives.
#
# Boundary vector ordering (m = 2(Ny-1) + 2(Nx-1) non-corner points):
#     left   : (1,     j),  j = 2:Ny         positions layout.left
#     right  : (Nx+1,  j),  j = 2:Ny         positions layout.right
#     bottom : (i,     1),  i = 2:Nx         positions layout.bottom
#     top    : (i,  Ny+1),  i = 2:Nx         positions layout.top
#
# Corners are deliberately excluded:
#   * the interior Helmholtz/Poisson stencils never touch a corner value
#     (Dx² acts on columns j∈2:Ny, Dy² on rows i∈2:Nx), so corner ω is decoupled;
#   * ψ|Γ = 0 forces ∂ₙψ = 0 at the corners identically (the derivative along a
#     wall of a field that vanishes on that wall is zero), so the corner
#     normal-derivative constraints are redundant;
#   * in the q-formulation the corner values of q never enter `laplacian(q)`
#     at any non-corner node (Wx[1] = Wy[1] = 0), and ω = laplacian(q) is
#     exactly zero at the corners.
# Including the corners would add 4 zero rows/columns to the influence matrix.
# ============================================================================

struct BoundaryLayout
    Nx::Int
    Ny::Int
    m::Int
    points::Vector{CartesianIndex{2}}     # grid index of boundary dof k
    wall::Vector{Symbol}                  # :left, :right, :bottom, :top
    normal_sign::Vector{Int}              # -1 for left/bottom, +1 for right/top
    left::UnitRange{Int}
    right::UnitRange{Int}
    bottom::UnitRange{Int}
    top::UnitRange{Int}
    corners::Vector{CartesianIndex{2}}
end

function BoundaryLayout(Nx::Int, Ny::Int)
    nL = Ny - 1; nB = Nx - 1
    left   = 1:nL
    right  = nL+1:2nL
    bottom = 2nL+1:2nL+nB
    top    = 2nL+nB+1:2nL+2nB
    m = 2nL + 2nB
    points = Vector{CartesianIndex{2}}(undef, m)
    wall = Vector{Symbol}(undef, m)
    nsg = Vector{Int}(undef, m)
    for (k, j) in enumerate(2:Ny)
        points[left[k]]  = CartesianIndex(1, j);    wall[left[k]]  = :left;  nsg[left[k]]  = -1
        points[right[k]] = CartesianIndex(Nx+1, j); wall[right[k]] = :right; nsg[right[k]] = +1
    end
    for (k, i) in enumerate(2:Nx)
        points[bottom[k]] = CartesianIndex(i, 1);    wall[bottom[k]] = :bottom; nsg[bottom[k]] = -1
        points[top[k]]    = CartesianIndex(i, Ny+1); wall[top[k]]    = :top;    nsg[top[k]]    = +1
    end
    corners = [CartesianIndex(1,1), CartesianIndex(Nx+1,1), CartesianIndex(1,Ny+1), CartesianIndex(Nx+1,Ny+1)]
    return BoundaryLayout(Nx, Ny, m, points, wall, nsg, left, right, bottom, top, corners)
end

BoundaryLayout(grid::ChebyshevGrid{T,2}) where {T} = BoundaryLayout(grid.Ns[1], grid.Ns[2])

"""
    gather!(v, M, layout)  /  gather(M, layout)

Boundary vector from the four edges of `M` (corners excluded).
"""
function gather!(v::AbstractVector, M::AbstractMatrix, L::BoundaryLayout)
    @inbounds for k in 1:L.m
        v[k] = M[L.points[k]]
    end
    return v
end
gather(M::AbstractMatrix{T}, L::BoundaryLayout) where {T} = gather!(Vector{T}(undef, L.m), M, L)

"""
    scatter!(M, v, layout; corners = zero(eltype(M)))

Write boundary vector `v` onto the four edges of `M`; corners set to `corners`.
Interior of `M` untouched.
"""
function scatter!(M::AbstractMatrix, v::AbstractVector, L::BoundaryLayout; corners = zero(eltype(M)))
    @inbounds for k in 1:L.m
        M[L.points[k]] = v[k]
    end
    @inbounds for c in L.corners
        M[c] = corners
    end
    return M
end

"""
    normal_derivative_boundary!(d, ψ, ops, layout)  /  normal_derivative_boundary(ψ, ops, layout)

Outward wall-normal derivative of the nodal field `ψ` using the plain
Chebyshev matrices:  left −ψ_x, right +ψ_x, bottom −ψ_y, top +ψ_y.
"""
function normal_derivative_boundary!(d::AbstractVector{T}, ψ::AbstractMatrix{T}, ops::CavityOperators{T}, L::BoundaryLayout) where {T}
    Nx, Ny = L.Nx, L.Ny
    Dx = ops.ψ.Dx; Dy = ops.ψ.Dy
    @inbounds for (k, j) in enumerate(2:Ny)
        sL = zero(T); sR = zero(T)
        for i in 1:Nx+1
            sL += Dx[1, i] * ψ[i, j]
            sR += Dx[Nx+1, i] * ψ[i, j]
        end
        d[L.left[k]] = -sL
        d[L.right[k]] = sR
    end
    @inbounds for (k, i) in enumerate(2:Nx)
        sB = zero(T); sT = zero(T)
        for j in 1:Ny+1
            sB += ψ[i, j] * Dy[1, j]
            sT += ψ[i, j] * Dy[Ny+1, j]
        end
        d[L.bottom[k]] = -sB
        d[L.top[k]] = sT
    end
    return d
end
normal_derivative_boundary(ψ::AbstractMatrix{T}, ops, L::BoundaryLayout) where {T} =
    normal_derivative_boundary!(Vector{T}(undef, L.m), ψ, ops, L)

"""
    normal_derivative_boundary_q(q, ops, layout)

Outward normal derivative of ψ = (1-x²)(1-y²)q in the *exact polynomial* sense
used by the reference `velocity(q, ops)`:

    ψ_y|_{y=±1} = ∓2 (1-x²) q(x, ±1),      ψ_x|_{x=±1} = ∓2 (1-y²) q(±1, y)

so  ∂ₙψ = −2 w q|Γ  on every wall (w = 1-x² along top/bottom, 1-y² along left/right),
i.e. the normal derivative is carried entirely by the boundary values of q.
"""
function normal_derivative_boundary_q(q::AbstractMatrix{T}, ops::CavityOperators{T}, L::BoundaryLayout) where {T}
    wx = ops.q.Wx.diag; wy = ops.q.Wy.diag
    d = Vector{T}(undef, L.m)
    @inbounds for (k, j) in enumerate(2:L.Ny)
        d[L.left[k]]  = -2 * wy[j] * q[1, j]
        d[L.right[k]] = -2 * wy[j] * q[L.Nx+1, j]
    end
    @inbounds for (k, i) in enumerate(2:L.Nx)
        d[L.bottom[k]] = -2 * wx[i] * q[i, 1]
        d[L.top[k]]    = -2 * wx[i] * q[i, L.Ny+1]
    end
    return d
end

"""
    lid_q_boundary(grid, layout) -> g

The reference Dirichlet data for q on the boundary, in layout order:
0 on left/right/bottom and −½(1−x²) on the top (exactly `apply_bcs_rhs!`).
"""
function lid_q_boundary(grid::ChebyshevGrid{T,2}, L::BoundaryLayout) where {T}
    x = grid.xs[1]
    g = zeros(T, L.m)
    @inbounds for (k, i) in enumerate(2:L.Nx)
        g[L.top[k]] = -1/2 * (1 - x[i]^2)
    end
    return g
end

"""
    lid_normal_derivative_target(grid, ops, layout) -> h

Target ∂ₙψ|Γ implied by the reference boundary data (polynomial sense):
top wall  ψ_y = −2(1−x²)·(−½(1−x²)) = (1−x²)²  (regularised lid, u(x,1) = (1−x²)²),
zero on the other three walls.  Derived by evaluating
`normal_derivative_boundary_q` on the reference boundary data, not hard-coded.
"""
function lid_normal_derivative_target(grid::ChebyshevGrid{T,2}, ops::CavityOperators{T}, L::BoundaryLayout) where {T}
    q = zeros(T, L.Nx+1, L.Ny+1)
    scatter!(q, lid_q_boundary(grid, L), L)
    return normal_derivative_boundary_q(q, ops, L)
end
