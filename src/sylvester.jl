# ============================================================================
# Phase 3 — separable (tensor-product) Dirichlet solvers.
#
# Every solver here reduces to an interior Sylvester equation
#
#       A X + X Bᵀ = G,        A ∈ R^{(Nx-1)×(Nx-1)},  B ∈ R^{(Ny-1)×(Ny-1)}
#
# which is solved with precomputed decompositions of the 1-D operators only.
# No (N+1)²×(N+1)² matrix is ever formed.
#
# Two backends:
#   :eigen  — A = VA ΛA VA⁻¹, B = VB ΛB VB⁻¹ (real eigenvalues required)
#             X = VA * ((VA⁻¹ G VB⁻ᵀ) ./ (λA_i + λB_j)) * VBᵀ          (4 GEMMs)
#   :schur  — A = ZA TA ZAᵀ, B = ZB TB ZBᵀ (real Schur, Bartels–Stewart)
#             TA Y + Y TBᵀ = ZAᵀ G ZB  (LAPACK trsyl),  X = ZA Y ZBᵀ  (4 GEMMs + trsyl)
#   :ceigen — same as :eigen but in complex arithmetic (for operators with a complex
#             spectrum, e.g. the q-form Wx⁻¹D²x_q); 4 complex GEMMs, result projected
#             to the real part.  Faster than :schur (trsyl is unblocked) but its
#             accuracy degrades with κ(V); use only where validated.
# ============================================================================

struct SylvesterSolver{T<:AbstractFloat}
    mode::Symbol
    nx::Int
    ny::Int
    # eigen mode
    VA::Matrix{T}; VAinv::Matrix{T}; λA::Vector{T}
    VB::Matrix{T}; VBinv::Matrix{T}; λB::Vector{T}
    invden::Matrix{T}                       # 1 / (λA_i + λB_j)
    # schur mode
    ZA::Matrix{T}; TA::Matrix{T}
    ZB::Matrix{T}; TB::Matrix{T}
    # complex eigen mode
    cVA::Matrix{Complex{T}}; cVAinv::Matrix{Complex{T}}
    cVB::Matrix{Complex{T}}; cVBinv::Matrix{Complex{T}}
    cinvden::Matrix{Complex{T}}
    cW1::Matrix{Complex{T}}; cW2::Matrix{Complex{T}}; cG::Matrix{Complex{T}}
    # work
    W1::Matrix{T}; W2::Matrix{T}
end

_noc(T, n, m) = zeros(Complex{T}, n, m)

"""
    SylvesterSolver(A, B; mode = :eigen)

Precomputes everything needed to solve `A X + X Bᵀ = G` repeatedly.
`mode = :eigen` requires `A` and `B` to be diagonalizable with real spectra
(checked; falls back to `:schur` with a warning otherwise).
"""
function SylvesterSolver(A::AbstractMatrix{T}, B::AbstractMatrix{T}; mode::Symbol = :eigen,
                         imag_tol = 1e-10) where {T<:AbstractFloat}
    nx, ny = size(A, 1), size(B, 1)
    A = Matrix{T}(A); B = Matrix{T}(B)
    if mode == :eigen
        EA = eigen(A); EB = eigen(B)
        if maximum(abs, imag.(EA.values); init = 0.0) > imag_tol * maximum(abs, EA.values) ||
           maximum(abs, imag.(EB.values); init = 0.0) > imag_tol * maximum(abs, EB.values)
            @info "SylvesterSolver: complex eigenvalues detected, falling back to :schur"
            mode = :schur
        else
            VA = real.(EA.vectors); λA = real.(EA.values)
            VB = real.(EB.vectors); λB = real.(EB.values)
            invden = [one(T) / (λA[i] + λB[j]) for i in 1:nx, j in 1:ny]
            any(!isfinite, invden) && error("SylvesterSolver: singular Sylvester operator (λA_i + λB_j = 0)")
            return SylvesterSolver{T}(:eigen, nx, ny, VA, inv(VA), λA, VB, inv(VB), λB, invden,
                                      zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
                                      _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0),
                                      zeros(T, nx, ny), zeros(T, nx, ny))
        end
    end
    if mode == :ceigen
        EA = eigen(A); EB = eigen(B)
        VA = Matrix{Complex{T}}(EA.vectors); VB = Matrix{Complex{T}}(EB.vectors)
        λA = Complex{T}.(EA.values); λB = Complex{T}.(EB.values)
        invden = [one(Complex{T}) / (λA[i] + λB[j]) for i in 1:nx, j in 1:ny]
        any(!isfinite, invden) && error("SylvesterSolver: singular Sylvester operator (λA_i + λB_j = 0)")
        return SylvesterSolver{T}(:ceigen, nx, ny,
                                  zeros(T, 0, 0), zeros(T, 0, 0), T[], zeros(T, 0, 0), zeros(T, 0, 0), T[], zeros(T, 0, 0),
                                  zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
                                  VA, inv(VA), VB, inv(VB), invden, _noc(T,nx,ny), _noc(T,nx,ny), _noc(T,nx,ny),
                                  zeros(T, nx, ny), zeros(T, nx, ny))
    end
    mode == :schur || error("unknown mode $mode")
    SA = schur(A); SB = schur(B)
    return SylvesterSolver{T}(:schur, nx, ny,
                              zeros(T, 0, 0), zeros(T, 0, 0), T[], zeros(T, 0, 0), zeros(T, 0, 0), T[], zeros(T, 0, 0),
                              Matrix(SA.Z), Matrix(SA.T), Matrix(SB.Z), Matrix(SB.T),
                              _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0), _noc(T,0,0),
                              zeros(T, nx, ny), zeros(T, nx, ny))
end

"""
    solve!(X, S::SylvesterSolver, G)

Solves `A X + X Bᵀ = G` in place (X may alias G). No allocations.
"""
function solve!(X::AbstractMatrix{T}, S::SylvesterSolver{T}, G::AbstractMatrix{T}) where {T}
    W1, W2 = S.W1, S.W2
    if S.mode == :eigen
        mul!(W1, S.VAinv, G)
        mul!(W2, W1, transpose(S.VBinv))
        W2 .*= S.invden
        mul!(W1, S.VA, W2)
        mul!(X, W1, transpose(S.VB))
    elseif S.mode == :ceigen
        S.cG .= G
        mul!(S.cW1, S.cVAinv, S.cG)
        mul!(S.cW2, S.cW1, transpose(S.cVBinv))
        S.cW2 .*= S.cinvden
        mul!(S.cW1, S.cVA, S.cW2)
        mul!(S.cG, S.cW1, transpose(S.cVB))
        X .= real.(S.cG)
    else
        mul!(W1, transpose(S.ZA), G)
        mul!(W2, W1, S.ZB)
        _, scale = LAPACK.trsyl!('N', 'T', S.TA, S.TB, W2)
        scale == one(T) || (W2 ./= scale)
        mul!(W1, S.ZA, W2)
        mul!(X, W1, transpose(S.ZB))
    end
    return X
end

# boundary rows ω[[1,Nx+1], 2:Ny] → Bx (2×(Ny-1)),  boundary cols ω[2:Nx, [1,Ny+1]] → By ((Nx-1)×2)
function _boundary_rows_cols!(Bx, By, ω, Nx, Ny)
    @inbounds for (k, j) in enumerate(2:Ny)
        Bx[1, k] = ω[1, j]; Bx[2, k] = ω[Nx+1, j]
    end
    @inbounds for (k, i) in enumerate(2:Nx)
        By[k, 1] = ω[i, 1]; By[k, 2] = ω[i, Ny+1]
    end
    return Bx, By
end

# ----------------------------------------------------------------------------
# Dirichlet problems on the full grid.  Fields are (Nx+1)×(Ny+1); the boundary
# rows/columns of the *solution array* are inputs (prescribed Dirichlet data)
# and the interior is overwritten with the solution.
# ----------------------------------------------------------------------------

"""
    SeparableHelmholtzSolver(ops, c; mode = :eigen)

Solves  (I − cΔ) ω = f  on the interior, ω|Γ prescribed, with Δ = Dx² ⊗ I + I ⊗ Dy²
(the plain Chebyshev Laplacian `ops.ψ`).  Interior equation:

    (½I − c Dx²ᵢᵢ) Ωᵢ + Ωᵢ (½I − c Dy²ⱼⱼ)ᵀ = fᵢ + c (Dx²ᵢ,ᵦ Ωᵦ,ⱼ + Ωᵢ,ᵦ Dy²ⱼ,ᵦᵀ)
"""
struct SeparableHelmholtzSolver{T<:AbstractFloat}
    Nx::Int; Ny::Int; c::T
    syl::SylvesterSolver{T}
    Dxx_ib::Matrix{T}     # Dx²[ii, [1, Nx+1]]   (Nx-1)×2
    Dyy_jb::Matrix{T}     # Dy²[jj, [1, Ny+1]]   (Ny-1)×2
    Bx::Matrix{T}         # 2×(Ny-1) work: boundary rows ω[[1,Nx+1], jj]
    By::Matrix{T}         # (Nx-1)×2 work: boundary cols ω[ii, [1,Ny+1]]
    G::Matrix{T}          # interior RHS work
    Xi::Matrix{T}         # interior solution work
end

function SeparableHelmholtzSolver(ops::CavityOperators{T}, c::T; mode::Symbol = :eigen) where {T}
    Nx = size(ops.ψ.Dx, 1) - 1; Ny = size(ops.ψ.Dy, 1) - 1
    ii = 2:Nx; jj = 2:Ny
    D2x = ops.ψ.D²x; D2y = ops.ψ.D²y
    A = Matrix{T}(I, Nx-1, Nx-1) ./ 2 .- c .* D2x[ii, ii]
    B = Matrix{T}(I, Ny-1, Ny-1) ./ 2 .- c .* D2y[jj, jj]
    syl = SylvesterSolver(A, B; mode)
    return SeparableHelmholtzSolver{T}(Nx, Ny, c, syl, D2x[ii, [1, Nx+1]], D2y[jj, [1, Ny+1]],
                                       zeros(T, 2, Ny-1), zeros(T, Nx-1, 2),
                                       zeros(T, Nx-1, Ny-1), zeros(T, Nx-1, Ny-1))
end

"""
    solve_helmholtz_dirichlet!(ω, f, H::SeparableHelmholtzSolver)

Boundary rows/columns of `ω` are the Dirichlet data (input); the interior of
`ω` is overwritten with the solution.  Only the interior of `f` is read.
"""
function solve_helmholtz_dirichlet!(ω::AbstractMatrix{T}, f::AbstractMatrix{T}, H::SeparableHelmholtzSolver{T}) where {T}
    Nx, Ny, c = H.Nx, H.Ny, H.c
    ii = 2:Nx; jj = 2:Ny
    G = H.G
    G .= view(f, ii, jj)
    _boundary_rows_cols!(H.Bx, H.By, ω, Nx, Ny)
    mul!(G, H.Dxx_ib, H.Bx, c, one(T))                 # + c Dx²[ii,b] ω[b,jj]
    mul!(G, H.By, transpose(H.Dyy_jb), c, one(T))      # + c ω[ii,b] Dy²[jj,b]ᵀ
    solve!(H.Xi, H.syl, G)
    view(ω, ii, jj) .= H.Xi
    return ω
end

"""
    SeparablePoissonSolver(ops; mode = :eigen)

Solves  Δψ = ω  (plain Chebyshev Laplacian, P_N streamfunction) with ψ|Γ prescribed.
"""
struct SeparablePoissonSolver{T<:AbstractFloat}
    Nx::Int; Ny::Int
    syl::SylvesterSolver{T}
    Dxx_ib::Matrix{T}; Dyy_jb::Matrix{T}
    Bx::Matrix{T}; By::Matrix{T}
    G::Matrix{T}; Xi::Matrix{T}
end

function SeparablePoissonSolver(ops::CavityOperators{T}; mode::Symbol = :eigen) where {T}
    Nx = size(ops.ψ.Dx, 1) - 1; Ny = size(ops.ψ.Dy, 1) - 1
    ii = 2:Nx; jj = 2:Ny
    D2x = ops.ψ.D²x; D2y = ops.ψ.D²y
    syl = SylvesterSolver(Matrix(D2x[ii, ii]), Matrix(D2y[jj, jj]); mode)
    return SeparablePoissonSolver{T}(Nx, Ny, syl, D2x[ii, [1, Nx+1]], D2y[jj, [1, Ny+1]],
                                     zeros(T, 2, Ny-1), zeros(T, Nx-1, 2),
                                     zeros(T, Nx-1, Ny-1), zeros(T, Nx-1, Ny-1))
end

"""
    solve_poisson_dirichlet!(ψ, ω, P::SeparablePoissonSolver)

Boundary of `ψ` = Dirichlet data (input); interior overwritten with the solution of
`Dx² ψ + ψ Dy²ᵀ = ω` at interior nodes.
"""
function solve_poisson_dirichlet!(ψ::AbstractMatrix{T}, ω::AbstractMatrix{T}, P::SeparablePoissonSolver{T}) where {T}
    Nx, Ny = P.Nx, P.Ny
    ii = 2:Nx; jj = 2:Ny
    G = P.G
    G .= view(ω, ii, jj)
    _boundary_rows_cols!(P.Bx, P.By, ψ, Nx, Ny)
    mul!(G, P.Dxx_ib, P.Bx, -one(T), one(T))
    mul!(G, P.By, transpose(P.Dyy_jb), -one(T), one(T))
    solve!(P.Xi, P.syl, G)
    view(ψ, ii, jj) .= P.Xi
    return ψ
end

"""
    SeparableQPoissonSolver(ops; mode = :schur)

The *reference-exact* "Poisson" stage: given ω = `laplacian(q, ops)` at interior
nodes and q|Γ prescribed, recover the interior of q.  The reference operator is

    L q = D²x_q q Wy + Wx q D²y_qᵀ        (Δ of ψ = (1-x²)(1-y²)q, exact polynomial derivative)

whose interior block becomes, after scaling by Wxᵢᵢ⁻¹ (left) and Wyⱼⱼ⁻¹ (right),

    (Wxᵢᵢ⁻¹ D²x_qᵢᵢ) Qᵢ + Qᵢ (Wyⱼⱼ⁻¹ D²y_qⱼⱼ)ᵀ = Wxᵢᵢ⁻¹ (ωᵢ − D²x_qᵢ,ᵦ Qᵦ,ⱼ Wyⱼⱼ − Wxᵢᵢ Qᵢ,ᵦ D²y_qⱼ,ᵦᵀ) Wyⱼⱼ⁻¹.

`Wx⁻¹D²x_q` has a complex spectrum, so the default backend is `:schur`.
"""
struct SeparableQPoissonSolver{T<:AbstractFloat}
    Nx::Int; Ny::Int
    syl::SylvesterSolver{T}
    Lx_ib::Matrix{T}      # D²x_q[ii, b]
    Ly_jb::Matrix{T}      # D²y_q[jj, b]
    wx_i::Vector{T}; wy_j::Vector{T}          # interior (1-x²), (1-y²)
    Bx::Matrix{T}; By::Matrix{T}
    G::Matrix{T}; Xi::Matrix{T}
end

function SeparableQPoissonSolver(ops::CavityOperators{T}; mode::Symbol = :schur) where {T}
    Nx = size(ops.ψ.Dx, 1) - 1; Ny = size(ops.ψ.Dy, 1) - 1
    ii = 2:Nx; jj = 2:Ny
    wx = ops.q.Wx.diag; wy = ops.q.Wy.diag
    A = Diagonal(1 ./ wx[ii]) * ops.q.D²x[ii, ii]
    B = Diagonal(1 ./ wy[jj]) * ops.q.D²y[jj, jj]
    syl = SylvesterSolver(Matrix(A), Matrix(B); mode)
    return SeparableQPoissonSolver{T}(Nx, Ny, syl, ops.q.D²x[ii, [1, Nx+1]], ops.q.D²y[jj, [1, Ny+1]],
                                      wx[ii], wy[jj], zeros(T, 2, Ny-1), zeros(T, Nx-1, 2),
                                      zeros(T, Nx-1, Ny-1), zeros(T, Nx-1, Ny-1))
end

"""
    solve_qpoisson_dirichlet!(q, ω, P::SeparableQPoissonSolver)

Boundary of `q` = Dirichlet data (input); interior of `q` overwritten so that
`laplacian(q, ops)` equals `ω` at every interior node.
"""
function solve_qpoisson_dirichlet!(q::AbstractMatrix{T}, ω::AbstractMatrix{T}, P::SeparableQPoissonSolver{T}) where {T}
    Nx, Ny = P.Nx, P.Ny
    ii = 2:Nx; jj = 2:Ny
    G = P.G
    G .= view(ω, ii, jj)
    _boundary_rows_cols!(P.Bx, P.By, q, Nx, Ny)
    mul!(P.Xi, P.Lx_ib, P.Bx)                     # D²x_q[ii,b] q[b,jj]
    G .-= P.Xi .* P.wy_j'                         #   … Wy_jj
    mul!(P.Xi, P.By, transpose(P.Ly_jb))          # q[ii,b] D²y_q[jj,b]ᵀ
    G .-= P.wx_i .* P.Xi                          # Wx_ii …
    G .= G ./ P.wx_i ./ P.wy_j'                   # scale by Wx⁻¹ (left), Wy⁻¹ (right)
    solve!(P.Xi, P.syl, G)
    view(q, ii, jj) .= P.Xi
    return q
end
