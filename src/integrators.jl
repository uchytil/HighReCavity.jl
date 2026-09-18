# Time integrators for the vorticity equation  ∂ₜω = ν Δω − N(q),   ω = L q,
# with the diffusion term implicit and the convection term explicit.
#
#   ARK3   — Kennedy & Carpenter ARK3(2)4L[2]SA (default): 4-stage, third-order additive
#            Runge–Kutta with an ESDIRK implicit part (single diagonal coefficient γ, L-stable,
#            stiffly accurate) and an explicit RK part sharing the weights b.  One-step.
#   CNAB2  — Crank–Nicolson / second-order Adams–Bashforth (two-step; state qⁿ, qⁿ⁻¹).
#
# Both use the same implicit operator (I − cΔ) with c = γ·ν·dt, γ = ½ for CNAB2 and
# γ = 0.4358… for ARK3, so one InfluenceSolver serves the whole step.

abstract type TimeIntegrator end

struct CNAB2 <: TimeIntegrator end

struct ARK3{T} <: TimeIntegrator
    E::NTuple{4, Matrix{T}}     # explicit stage derivatives  −N(q_i)
    I::NTuple{4, Matrix{T}}     # implicit stage derivatives  ν Δ ω_i  (interior values)
    ωn::Matrix{T}               # ωⁿ = L qⁿ
end
ARK3{T}(n) where {T} = ARK3{T}(ntuple(_ -> zeros(T, n, n), 4), ntuple(_ -> zeros(T, n, n), 4), zeros(T, n, n))

integrator_type(::Val{:cnab2}, ::Type{T}, n) where {T} = CNAB2()
integrator_type(::Val{:ark3},  ::Type{T}, n) where {T} = ARK3{T}(n)

"Implicit weight γ of the scheme: the implicit solve uses c = γ ν dt."
implicit_weight(::CNAB2) = 1/2
implicit_weight(::ARK3) = ARK3_γ

# ---- ARK3(2)4L[2]SA tableau (Kennedy & Carpenter, Appl. Numer. Math. 44, 2003) ----------
const ARK3_γ = 1767732205903/4055673282236
const ARK3_AE = (                                   # explicit part, rows 2..4
    (1767732205903/2027836641118,),
    (5535828885825/10492691773637, 788022342437/10882634858940),
    (6485989280629/16251701735622, -4246266847089/9704473918619, 10755448449292/10357097424841))
const ARK3_AI = (                                   # implicit part, off-diagonal entries of rows 2..4 (diagonal = γ)
    (ARK3_γ,),
    (2746238789719/10658868560708, -640167445237/6845629431997),
    (1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821))
const ARK3_b = (1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821, ARK3_γ)
const ARK3_bhat = (2756255671327/12835298489170, -10771552573575/22201958757719, 9247589265047/10645013368117, 2193209047091/5459859503100)

"""
    ark3_step!(sim, s::ARK3)

One ARK3(2)4L[2]SA step.  Stage i solves  (I − γ ν dt Δ) ω_i = ωⁿ + dt Σ_{j<i} (a^E_ij E_j + a^I_ij I_j)
through the influence solver, which also yields q_i; the implicit derivative is recovered
algebraically, I_i = (ω_i − f_i)/(γ dt), and E_i = −N(q_i).  Since only the implicit part is
stiffly accurate, the step ends with  ω^{n+1} = ω₄ + dt Σ_j (b_j − a^E_4j) E_j  at interior nodes
and a q-Poisson solve that recovers q^{n+1} (and the consistent wall vorticity) from it.
"""
function ark3_step!(sim, s::ARK3)
    p = sim.params; ops = sim.ops; W = sim.work; f = sim.f
    dt = p.dt; ν = 1 / reynolds_internal(p); γdt = ARK3_γ * dt
    E, I = s.E, s.I
    # stage 1 (explicit): q₁ = qⁿ
    sim.q_prev .= sim.q
    laplacian!(s.ωn, sim.q, ops, W.A)                       # ωⁿ = L qⁿ, wall values included
    convection!(E[1], sim.q, ops, W); E[1] .*= -1
    laplacian_nodal!(I[1], s.ωn, ops); I[1] .*= ν            # ν Δ ωⁿ
    # stages 2..4 (implicit)
    for i in 2:4
        f .= s.ωn
        for j in 1:i-1
            f .+= (dt * ARK3_AE[i-1][j]) .* E[j] .+ (dt * ARK3_AI[i-1][j]) .* I[j]
        end
        influence_solve!(sim.q, sim.ω, f, sim.solver)       # (I − γ ν dt Δ) ω_i = f,  q_i
        I[i] .= (sim.ω .- f) ./ γdt                          # ν Δ ω_i  (valid at interior nodes)
        convection!(E[i], sim.q, ops, W); E[i] .*= -1
    end
    # final combination (interior) and projection onto the state q
    for j in 1:4
        w = ARK3_b[j] - (j < 4 ? ARK3_AE[3][j] : 0.0)
        sim.ω .+= (dt * w) .* E[j]
    end
    set_walls!(sim.q, sim.solver.g, sim.solver.layout)
    qpoisson!(sim.q, sim.ω, sim.solver.pois)                 # L q^{n+1} = ω^{n+1} (interior), q|Γ = g
    laplacian!(sim.ω, sim.q, ops, W.A)                       # ω^{n+1} = L q^{n+1} at all nodes
    return sim
end
