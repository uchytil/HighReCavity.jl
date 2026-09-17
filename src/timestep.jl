# ============================================================================
# Time integration (AB2 convection / Crank–Nicolson diffusion), kept separate
# from the linear influence solve.
# ============================================================================

"""
    QWorkspace(ops)

Preallocated (Nx+1)×(Ny+1) work arrays for the in-place q-form right-hand side.
"""
struct QWorkspace{T}
    W1::Matrix{T}; W2::Matrix{T}; W3::Matrix{T}
    u::Matrix{T}; v::Matrix{T}; ω::Matrix{T}
    N::Matrix{T}; Nprev::Matrix{T}
    wx::Vector{T}; wy::Vector{T}
end
function QWorkspace(ops::CavityOperators{T}) where {T}
    n1 = size(ops.ψ.Dx, 1); n2 = size(ops.ψ.Dy, 1)
    z() = zeros(T, n1, n2)
    return QWorkspace{T}(z(), z(), z(), z(), z(), z(), z(), z(), ops.q.Wx.diag, ops.q.Wy.diag)
end

"""
    laplacian!(ω, q, ops, W)   — in-place `laplacian(q, ops)` = Wx q D²y_qᵀ + D²x_q q Wy
"""
function laplacian!(ω, q, ops::CavityOperators, W::QWorkspace)
    mul!(W.W1, q, transpose(ops.q.D²y))
    ω .= W.wx .* W.W1
    mul!(W.W1, ops.q.D²x, q)
    ω .+= W.W1 .* W.wy'
    return ω
end

"""
    biharmonic!(B, q, ops, W)  — in-place `biharmonic(q, ops)` (same three terms as the reference)
"""
function biharmonic!(B, q, ops::CavityOperators, W::QWorkspace)
    # term1 = Wx q D⁴y_qᵀ
    mul!(W.W1, q, transpose(ops.q.D⁴y))
    B .= W.wx .* W.W1
    # term2 = D⁴x_q q Wy
    mul!(W.W1, ops.q.D⁴x, q)
    B .+= W.W1 .* W.wy'
    # term3 = D²x (Wx q D²y_qᵀ) + (D²x_q q Wy) D²yᵀ
    mul!(W.W1, q, transpose(ops.q.D²y)); W.W2 .= W.wx .* W.W1
    mul!(W.W1, ops.ψ.D²x, W.W2); B .+= W.W1
    mul!(W.W1, ops.q.D²x, q);  W.W2 .= W.W1 .* W.wy'
    mul!(W.W1, W.W2, transpose(ops.ψ.D²y)); B .+= W.W1
    return B
end

"""
    convection!(N, q, ops, W)  — in-place `convection(q, ops)` = u⊙ω_x + v⊙ω_y
"""
function convection!(N, q, ops::CavityOperators, W::QWorkspace)
    mul!(W.W1, q, transpose(ops.q.Dy)); W.u .= W.wx .* W.W1             # u = Wx q Dy_qᵀ
    mul!(W.W1, ops.q.Dx, q);            W.v .= .-(W.W1 .* W.wy')        # v = −Dx_q q Wy
    laplacian!(W.ω, q, ops, W)
    mul!(W.W1, ops.ψ.Dx, W.ω)                                           # ω_x
    mul!(W.W2, W.ω, transpose(ops.ψ.Dy))                                # ω_y
    N .= W.u .* W.W1 .+ W.v .* W.W2
    return N
end

"""
    qform_rhs!(rhs, q, q_prev, ops, Δt, Re, W)

The reference explicit RHS (interior values are what matters):
    laplacian(q) + ½Δt/Re·biharmonic(q) − Δt(3/2 N(q) − 1/2 N(q_prev)).
"""
function qform_rhs!(rhs, q, q_prev, ops::CavityOperators, Δt, Re, W::QWorkspace)
    laplacian!(rhs, q, ops, W)
    biharmonic!(W.W3, q, ops, W)
    rhs .+= (0.5 * Δt / Re) .* W.W3
    convection!(W.N, q, ops, W)
    convection!(W.Nprev, q_prev, ops, W)
    rhs .-= Δt .* (3/2 .* W.N .- 1/2 .* W.Nprev)
    return rhs
end

"""
    QState(sys::InfluenceSolver{QForm})  — q, q_prev, ω, rhs and workspace for `step_influence!`
"""
mutable struct QState{T}
    q::Matrix{T}
    q_prev::Matrix{T}
    ω::Matrix{T}
    rhs::Matrix{T}
    W::QWorkspace{T}
end
function QState(S::InfluenceSolver{QForm,T}) where {T}
    Nx, Ny = S.grid.Ns
    # Same initial state as the reference script: q_prev is copied *before* the
    # lid data is imposed on q, so q_prev = 0 on the first step.
    q = zeros(T, Nx+1, Ny+1)
    q_prev = copy(q)
    scatter!(q, S.g, S.layout)
    return QState{T}(q, q_prev, zeros(T, Nx+1, Ny+1), zeros(T, Nx+1, Ny+1), QWorkspace(S.ops))
end

"""
    step_influence!(st::QState, S::InfluenceSolver{QForm}; method=:resolve)

One AB2/CN step identical in exact arithmetic to `step_reference!`, but with the
dense inverse replaced by the influence-matrix solve.  Also leaves the new
vorticity in `st.ω`.
"""
function step_influence!(st::QState, S::InfluenceSolver{QForm}; method::Symbol = :resolve)
    qform_rhs!(st.rhs, st.q, st.q_prev, S.ops, S.Δt, S.Re, st.W)
    st.q_prev .= st.q
    solve_timestep_linear!(st.ω, st.q, st.rhs, S; method)
    return st
end

# ----------------------------------------------------------------------------
# Textbook ψ–ω stepping (state = ψ, ω, previous convection term)
# ----------------------------------------------------------------------------
mutable struct PsiOmegaState{T}
    ψ::Matrix{T}
    ω::Matrix{T}
    N::Matrix{T}
    Nprev::Matrix{T}
    rhs::Matrix{T}
    u::Matrix{T}; v::Matrix{T}; ωx::Matrix{T}; ωy::Matrix{T}; W1::Matrix{T}
end
function PsiOmegaState(S::InfluenceSolver{PsiOmegaForm,T}) where {T}
    Nx, Ny = S.grid.Ns
    z() = zeros(T, Nx+1, Ny+1)
    return PsiOmegaState{T}(z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end

"""
    psiomega_rhs!(rhs, st, ops, Δt, Re)  —  f = ω + cΔω − Δt(3/2 N − 1/2 N_prev),  N = u⊙ω_x + v⊙ω_y
(`st.N` is overwritten with the current convection term; `st.Nprev` must hold the previous one)
"""
function psiomega_rhs!(rhs, st::PsiOmegaState, ops::CavityOperators, Δt, Re)
    c = 0.5 * Δt / Re
    laplacian_ω!(st.W1, st.ω, ops)
    rhs .= st.ω .+ c .* st.W1
    velocity_from_streamfunction!(st.u, st.v, st.ψ, ops)
    vorticity_derivatives!(st.ωx, st.ωy, st.ω, ops)
    st.N .= st.u .* st.ωx .+ st.v .* st.ωy
    rhs .-= Δt .* (3/2 .* st.N .- 1/2 .* st.Nprev)
    return rhs
end

function step_psiomega!(st::PsiOmegaState, S::InfluenceSolver{PsiOmegaForm}; method::Symbol = :resolve)
    psiomega_rhs!(st.rhs, st, S.ops, S.Δt, S.Re)
    st.Nprev .= st.N
    solve_timestep_linear!(st.ω, st.ψ, st.rhs, S; method)
    return st
end
