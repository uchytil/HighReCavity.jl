# ============================================================================
# Phase 2 — routines operating directly on the physical streamfunction ψ and
# vorticity ω (nodal values on the (Nx+1)×(Ny+1) Chebyshev grid).
#
# Conventions (identical to the reference code):
#   * a field F is an (Nx+1)×(Ny+1) matrix, F[i, j] = F(x_i, y_j)
#   * x-derivatives act from the left:   ∂x F = Dx * F
#   * y-derivatives act from the right:  ∂y F = F * Dy'
#   * u = ψ_y,  v = -ψ_x,  ω = Δψ,  N(ψ,ω) = u ⊙ ω_x + v ⊙ ω_y
#
# All derivative matrices are the *plain* Chebyshev matrices `ops.ψ.*`.  Note
# that on the nodal streamfunction Ψ = Wx q Wy these routines differ from the
# q-based reference routines by the P_N ↔ (1-x²)(1-y²)P_N aliasing discussed in
# notes/technical_note.md; they agree to spectral accuracy on resolved fields.
# ============================================================================

"""
    streamfunction_from_q(q, ops) -> Ψ = Wx q Wy
"""
streamfunction_from_q(q::AbstractMatrix, ops::CavityOperators) = ops.q.Wx * q * ops.q.Wy

"""
    velocity_from_streamfunction(ψ, ops) -> (u, v)   with  u = ψ Dyᵀ,  v = -Dx ψ
"""
function velocity_from_streamfunction(ψ::AbstractMatrix, ops::CavityOperators)
    u = ψ * ops.ψ.Dy'
    v = -(ops.ψ.Dx * ψ)
    return u, v
end

function velocity_from_streamfunction!(u, v, ψ, ops::CavityOperators)
    mul!(u, ψ, ops.ψ.Dy')
    mul!(v, ops.ψ.Dx, ψ, -one(eltype(v)), zero(eltype(v)))
    return u, v
end

"""
    vorticity_derivatives(ω, ops) -> (ω_x, ω_y)   with  ω_x = Dx ω,  ω_y = ω Dyᵀ
"""
function vorticity_derivatives(ω::AbstractMatrix, ops::CavityOperators)
    return ops.ψ.Dx * ω, ω * ops.ψ.Dy'
end

function vorticity_derivatives!(ωx, ωy, ω, ops::CavityOperators)
    mul!(ωx, ops.ψ.Dx, ω)
    mul!(ωy, ω, ops.ψ.Dy')
    return ωx, ωy
end

"""
    convection_from_uvω(u, v, ω, ops) -> u ⊙ (Dx ω) + v ⊙ (ω Dyᵀ)

The nonlinear term from *given* velocities.  With `u, v = velocity(q, ops)` and
`ω = laplacian(q, ops)` this reproduces the reference `convection(q, ops)`
to roundoff (same operations).
"""
function convection_from_uvω(u::AbstractMatrix, v::AbstractMatrix, ω::AbstractMatrix, ops::CavityOperators)
    ωx, ωy = vorticity_derivatives(ω, ops)
    return u .* ωx .+ v .* ωy
end

"""
    convection_from_ψω(ψ, ω, ops) -> N(ψ, ω) = ψ_y ⊙ ω_x − ψ_x ⊙ ω_y
"""
function convection_from_ψω(ψ::AbstractMatrix, ω::AbstractMatrix, ops::CavityOperators)
    u, v = velocity_from_streamfunction(ψ, ops)
    return convection_from_uvω(u, v, ω, ops)
end

"""
    laplacian_ψ(ψ, ops) -> Dx² ψ + ψ (Dy²)ᵀ
"""
laplacian_ψ(ψ::AbstractMatrix, ops::CavityOperators) = ops.ψ.D²x * ψ + ψ * ops.ψ.D²y'

"""
    laplacian_ω(ω, ops) -> Dx² ω + ω (Dy²)ᵀ    (same operator as `laplacian_ψ`)
"""
laplacian_ω(ω::AbstractMatrix, ops::CavityOperators) = laplacian_ψ(ω, ops)

function laplacian_ψ!(out, ψ, ops::CavityOperators)
    mul!(out, ops.ψ.D²x, ψ)
    mul!(out, ψ, ops.ψ.D²y', true, true)
    return out
end
const laplacian_ω! = laplacian_ψ!

"""
    laplacian_boundary!(out, q, ops, layout)

Boundary (non-corner) values of the reference `laplacian(q, ops)` = Δψ for
ψ = (1-x²)(1-y²)q, written into `out` in `BoundaryLayout` order.  Uses only the
four boundary rows/columns of the q-operators (O(N²) work).
"""
function laplacian_boundary!(out::AbstractVector, q::AbstractMatrix, ops::CavityOperators, layout)
    Nx, Ny = layout.Nx, layout.Ny
    D²x_q = ops.q.D²x; D²y_q = ops.q.D²y
    wx = ops.q.Wx.diag; wy = ops.q.Wy.diag
    # ω = D²x_q q Wy + Wx q D²y_qᵀ.  On the left/right walls Wx = 0 (x = ∓1) so
    # only the first term survives; on bottom/top only the second one does.
    @inbounds for (k, j) in enumerate(2:Ny)
        sL = zero(eltype(out)); sR = zero(eltype(out))
        for i in 1:Nx+1
            sL += D²x_q[1, i] * q[i, j]
            sR += D²x_q[Nx+1, i] * q[i, j]
        end
        out[layout.left[k]]  = sL * wy[j] + wx[1]    * dot(view(q, 1, :), view(D²y_q, j, :))
        out[layout.right[k]] = sR * wy[j] + wx[Nx+1] * dot(view(q, Nx+1, :), view(D²y_q, j, :))
    end
    @inbounds for (k, i) in enumerate(2:Nx)
        sB = zero(eltype(out)); sT = zero(eltype(out))
        for j in 1:Ny+1
            sB += q[i, j] * D²y_q[1, j]
            sT += q[i, j] * D²y_q[Ny+1, j]
        end
        out[layout.bottom[k]] = wx[i] * sB + wy[1]    * dot(view(D²x_q, i, :), view(q, :, 1))
        out[layout.top[k]]    = wx[i] * sT + wy[Ny+1] * dot(view(D²x_q, i, :), view(q, :, Ny+1))
    end
    return out
end
