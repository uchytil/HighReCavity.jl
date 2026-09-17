# Interior Dirichlet solves as Sylvester equations  A X + X Bᵀ = G  on the (N−1)² interior
# nodes, using precomputed decompositions of the two 1-D operators only.  Nothing of size
# (N+1)²×(N+1)² is ever formed.
#
# The cavity is square, so the same 1-D operator A appears on both sides:  A X + X Aᵀ = G.
#
#   :ceigen  A = V Λ V⁻¹ (eigendecomposition):
#            X = V [ (V⁻¹ G V⁻ᵀ) ./ (λ_i + λ_j) ] Vᵀ                    4 GEMMs
#            Real arithmetic when the spectrum is real (Helmholtz stage: ½I − cD2, κ(V) ≈ 2),
#            complex otherwise (q-Poisson stage: W⁻¹D2q has spurious complex high modes).
#   :schur   A = Z T Zᵀ (real Schur, Bartels–Stewart):
#            T Ŷ + Ŷ Tᵀ = Zᵀ G Z  (LAPACK trsyl),  X = Z Ŷ Zᵀ           4 GEMMs + trsyl
#            Backward stable regardless of κ(V); ~4× slower because trsyl is unblocked.

struct SylvesterSolver{T<:AbstractFloat, S<:Union{T, Complex{T}}}
    backend::Symbol
    # :ceigen  (S = T or Complex{T})
    V::Matrix{S}; Vinv::Matrix{S}
    invden::Matrix{S}                 # 1 / (λ_i + λ_j)
    C1::Matrix{S}; C2::Matrix{S}
    # :schur
    Z::Matrix{T}; Tm::Matrix{T}
    R1::Matrix{T}; R2::Matrix{T}
end

"Precompute the decomposition of A for solving  A X + X Aᵀ = G."
function SylvesterSolver(A::AbstractMatrix{T}, backend::Symbol) where {T<:AbstractFloat}
    n = size(A, 1)
    rz(k, l) = zeros(T, k, l)
    if backend == :ceigen
        E = eigen(Matrix(A)); λ = E.values
        S = eltype(λ) <: Real ? T : Complex{T}          # real spectrum → real arithmetic
        V = Matrix{S}(E.vectors)
        invden = [one(S) / (λ[i] + λ[j]) for i in 1:n, j in 1:n]
        all(isfinite, invden) || error("singular Sylvester operator (λ_i + λ_j = 0)")
        return SylvesterSolver{T,S}(backend, V, inv(V), invden, zeros(S, n, n), zeros(S, n, n), rz(0, 0), rz(0, 0), rz(0, 0), rz(0, 0))
    elseif backend == :schur
        F = schur(Matrix(A))
        return SylvesterSolver{T,T}(backend, rz(0, 0), rz(0, 0), rz(0, 0), rz(0, 0), rz(0, 0),
                                    Matrix(F.Z), Matrix(F.T), rz(n, n), rz(n, n))
    end
    error("unknown backend $backend (use :ceigen or :schur)")
end

"Solve A X + X Aᵀ = G in place (allocation-free)."
function sylvester_solve!(X::AbstractMatrix{T}, S::SylvesterSolver{T}, G::AbstractMatrix{T}) where {T}
    if S.backend == :ceigen
        S.C1 .= G
        mul!(S.C2, S.Vinv, S.C1)
        mul!(S.C1, S.C2, transpose(S.Vinv))
        S.C1 .*= S.invden
        mul!(S.C2, S.V, S.C1)
        mul!(S.C1, S.C2, transpose(S.V))
        X .= real.(S.C1)
    else
        mul!(S.R1, transpose(S.Z), G)
        mul!(S.R2, S.R1, S.Z)
        _, scale = LAPACK.trsyl!('N', 'T', S.Tm, S.Tm, S.R2)
        scale == one(T) || (S.R2 ./= scale)
        mul!(S.R1, S.Z, S.R2)
        mul!(X, S.R1, transpose(S.Z))
    end
    return X
end

# ---------------------------------------------------------------------------------------
# The two interior Dirichlet problems of one time step.  Fields are (N+1)×(N+1); the
# boundary rows/columns of the solution array are *inputs* (Dirichlet data) and the
# interior is overwritten.  ii = 2:N are interior indices, b = [1, N+1] the wall indices.
# ---------------------------------------------------------------------------------------

"""
Helmholtz stage:  (I − cΔ) ω = f  at interior nodes, Δ = D2·Ω + Ω·D2ᵀ, ω|Γ prescribed.

    (½I − c D2ᵢᵢ) Ωᵢ + Ωᵢ (½I − c D2ᵢᵢ)ᵀ = fᵢ + c (D2ᵢ,ᵦ Ωᵦ,ᵢ + Ωᵢ,ᵦ D2ᵢ,ᵦᵀ)
"""
struct HelmholtzSolver{T<:AbstractFloat, SY<:SylvesterSolver{T}}
    N::Int; c::T
    syl::SY
    D2_ib::Matrix{T}                 # D2[ii, b]      (N−1)×2
    Bx::Matrix{T}; By::Matrix{T}     # wall rows Ω[b, ii] (2×(N−1)) and wall columns Ω[ii, b] ((N−1)×2)
    G::Matrix{T}; Xi::Matrix{T}      # interior work
end

# The Helmholtz operator has a real, well-conditioned spectrum, so it always uses the eigen path
# (as in the validated implementation); `backend` only selects the q-Poisson stage.
function HelmholtzSolver(ops::CavityOperators{T}, c::T) where {T}
    N = ops.grid.N; ii = 2:N
    A = Matrix{T}(I, N-1, N-1) ./ 2 .- c .* ops.D2[ii, ii]
    syl = SylvesterSolver(A, :ceigen)
    return HelmholtzSolver{T,typeof(syl)}(N, c, syl, ops.D2[ii, [1, N+1]],
                                          zeros(T, 2, N-1), zeros(T, N-1, 2), zeros(T, N-1, N-1), zeros(T, N-1, N-1))
end

function helmholtz!(ω::AbstractMatrix{T}, f::AbstractMatrix{T}, H::HelmholtzSolver{T}) where {T}
    N = H.N; ii = 2:N
    wall_rows_cols!(H.Bx, H.By, ω, N)
    H.G .= view(f, ii, ii)
    mul!(H.G, H.D2_ib, H.Bx, H.c, one(T))
    mul!(H.G, H.By, transpose(H.D2_ib), H.c, one(T))
    sylvester_solve!(H.Xi, H.syl, H.G)
    view(ω, ii, ii) .= H.Xi
    return ω
end

"""
q-Poisson stage:  recover the interior of q from  ω = Δψ = D2q·Q·W + W·Q·D2qᵀ  at interior
nodes with q|Γ prescribed.  Scaling by Wᵢᵢ⁻¹ on both sides gives the Sylvester form

    (W⁻¹D2q)ᵢᵢ Qᵢ + Qᵢ (W⁻¹D2q)ᵢᵢᵀ = Wᵢᵢ⁻¹ (ωᵢ − D2qᵢ,ᵦ Qᵦ,ᵢ Wᵢᵢ − Wᵢᵢ Qᵢ,ᵦ D2qᵢ,ᵦᵀ) Wᵢᵢ⁻¹.

W⁻¹D2q has a complex spectrum (spurious high modes, κ(V) ~ 1e3), so `backend` matters here:
:ceigen (complex GEMMs, fast) or :schur (backward stable).
"""
struct QPoissonSolver{T<:AbstractFloat, SY<:SylvesterSolver{T}}
    N::Int
    syl::SY
    D2q_ib::Matrix{T}                # D2q[ii, b]
    wi::Vector{T}                    # interior 1 − x²
    Bx::Matrix{T}; By::Matrix{T}
    G::Matrix{T}; Xi::Matrix{T}
end

function QPoissonSolver(ops::CavityOperators{T}, backend::Symbol) where {T}
    N = ops.grid.N; ii = 2:N
    A = Diagonal(1 ./ ops.w[ii]) * ops.D2q[ii, ii]
    syl = SylvesterSolver(A, backend)
    return QPoissonSolver{T,typeof(syl)}(N, syl, ops.D2q[ii, [1, N+1]], ops.w[ii],
                                         zeros(T, 2, N-1), zeros(T, N-1, 2), zeros(T, N-1, N-1), zeros(T, N-1, N-1))
end

function qpoisson!(q::AbstractMatrix{T}, ω::AbstractMatrix{T}, P::QPoissonSolver{T}) where {T}
    N = P.N; ii = 2:N; wi = P.wi
    wall_rows_cols!(P.Bx, P.By, q, N)
    P.G .= view(ω, ii, ii)
    mul!(P.Xi, P.D2q_ib, P.Bx);              P.G .-= P.Xi .* wi'       # − D2q[ii,b] Q[b,ii] W
    mul!(P.Xi, P.By, transpose(P.D2q_ib));   P.G .-= wi .* P.Xi        # − W Q[ii,b] D2q[ii,b]ᵀ
    P.G .= P.G ./ wi ./ wi'
    sylvester_solve!(P.Xi, P.syl, P.G)
    view(q, ii, ii) .= P.Xi
    return q
end

# wall rows M[[1,N+1], 2:N] → Bx,  wall columns M[2:N, [1,N+1]] → By
function wall_rows_cols!(Bx, By, M, N)
    @inbounds for (k, j) in enumerate(2:N)
        Bx[1, k] = M[1, j];  Bx[2, k] = M[N+1, j]
        By[k, 1] = M[j, 1];  By[k, 2] = M[j, N+1]
    end
    return Bx, By
end
