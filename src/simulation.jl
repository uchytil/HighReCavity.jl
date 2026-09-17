# User-facing simulation: parameters, state, one time step, long runs.
#
# Time integration (unchanged from the original solver): Crank–Nicolson for diffusion,
# second-order Adams–Bashforth for convection, on the vorticity equation written for ω = Δψ:
#
#     (I − cΔ) ω^{n+1} = f,    f = ω^n + c Δω^n − dt (3/2 N^n − 1/2 N^{n−1}),    c = dt/(2 Re_internal)
#
# with ω^n = Δψ^n and Δω^n both formed from q^n by the exact product-rule operators
# (`laplacian!`, `biharmonic!`) and N = u ω_x + v ω_y (`convection!`).  The first step uses
# q^{−1} = 0 (so N^{−1} = 0), as the original script does.

"""
    CavityParameters(; N, Re, dt, alpha = 0.96, backend = :ceigen, T = Float64)

`Re` is the Reynolds number based on the **full cavity side length L = 2** (the cavity is
[-1, 1]², the lid speed is max (1−x²)² = 1).  The discrete operators are written in units of
the half-width, so internally the solver uses `Re_internal = Re / 2`; e.g. `Re = 30_000`
reproduces the original script's `Re = 30000 / 2`.
"""
struct CavityParameters{T<:AbstractFloat}
    N::Int
    Re::T             # public: based on the full side length L = 2
    dt::T
    alpha::T
    backend::Symbol   # q-Poisson stage: :ceigen (default, fast) or :schur (backward-stable reference backend)
end

function CavityParameters(; N::Int, Re, dt, alpha = 0.96, backend::Symbol = :ceigen, T::Type{<:AbstractFloat} = Float64)
    backend in (:ceigen, :schur) || throw(ArgumentError("backend must be :ceigen or :schur"))
    return CavityParameters{T}(N, T(Re), T(dt), T(alpha), backend)
end

"Reynolds number in the solver's own units (length scale = half-width 1): Re_internal = Re / 2."
reynolds_internal(p::CavityParameters) = p.Re / 2

"Diffusion coefficient of the implicit step, c = dt / (2 Re_internal)."
diffusion_coefficient(p::CavityParameters) = p.dt / (2 * reynolds_internal(p))

struct Workspace{T}
    A::Matrix{T}; B::Matrix{T}; C::Matrix{T}     # scratch for laplacian!/biharmonic!/convection!
    u::Matrix{T}; v::Matrix{T}
    Nn::Matrix{T}; Nprev::Matrix{T}              # convection terms N(qⁿ), N(qⁿ⁻¹)
end
Workspace{T}(n) where {T} = Workspace{T}(ntuple(_ -> zeros(T, n, n), 7)...)

mutable struct CavitySimulation{T<:AbstractFloat, S<:InfluenceSolver{T}}
    params::CavityParameters{T}
    ops::CavityOperators{T}
    solver::S
    q::Matrix{T}          # current state qⁿ  (ψ = (1−x²)(1−y²) q)
    q_prev::Matrix{T}     # qⁿ⁻¹ (for Adams–Bashforth)
    ω::Matrix{T}          # vorticity Δψⁿ (by-product of the last solve; zero before the first step)
    f::Matrix{T}          # right-hand side work array
    work::Workspace{T}
    step::Int
    t::T
end

"""
    CavitySimulation(params) 

Initial state: fluid at rest with the lid data imposed on the walls of q (the impulsively
started lid of the original script).  All fixed work (operators, decompositions, influence
matrix) happens here.
"""
function CavitySimulation(p::CavityParameters{T}) where {T}
    grid = ChebyshevGrid(p.N, p.alpha)
    ops = CavityOperators(grid)
    solver = InfluenceSolver(ops, diffusion_coefficient(p); backend = p.backend)
    n = p.N + 1
    q = zeros(T, n, n)
    q_prev = copy(q)                                   # q⁻¹ = 0 (before the lid data is imposed)
    set_walls!(q, solver.g, solver.layout)
    return CavitySimulation{T,typeof(solver)}(p, ops, solver, q, q_prev, zeros(T, n, n), zeros(T, n, n), Workspace{T}(n), 0, zero(T))
end

"""
    rhs!(f, q, q_prev, ops, dt, c, W)  —  f = Δψ + c Δ²ψ − dt (3/2 N(q) − 1/2 N(q_prev))
"""
function rhs!(f, q, q_prev, ops::CavityOperators, dt, c, W::Workspace)
    laplacian!(f, q, ops, W.A)                         # ωⁿ
    biharmonic!(W.B, q, ops, W.A, W.C);  f .+= c .* W.B  # + c Δωⁿ
    convection!(W.Nn, q, ops, W)
    convection!(W.Nprev, q_prev, ops, W)
    f .-= dt .* (3/2 .* W.Nn .- 1/2 .* W.Nprev)
    return f
end

"""
    step!(sim)  —  (qⁿ, qⁿ⁻¹) → f → influence solve → qⁿ⁺¹
"""
function step!(sim::CavitySimulation)
    p = sim.params
    rhs!(sim.f, sim.q, sim.q_prev, sim.ops, p.dt, diffusion_coefficient(p), sim.work)
    sim.q_prev .= sim.q
    influence_solve!(sim.q, sim.ω, sim.f, sim.solver)
    sim.step += 1
    sim.t += p.dt
    return sim
end

"""
    run!(sim, nsteps; callback = nothing, every = 1, check_nan = true)

Advance `nsteps` steps.  `callback(sim)` is called after every `every`-th step (and after
the last one); use it to save/inspect/process the state — nothing is stored otherwise.
"""
function run!(sim::CavitySimulation, nsteps::Integer; callback = nothing, every::Integer = 1, check_nan::Bool = true)
    for n in 1:nsteps
        step!(sim)
        check_nan && any(isnan, sim.q) && error("solution diverged at step $(sim.step) (t = $(sim.t))")
        if callback !== nothing && (n % every == 0 || n == nsteps)
            callback(sim)
        end
    end
    return sim
end

# ---------------------------------------------------------------------------------------
# Physical fields (same discrete definitions as the original q formulation)
# ---------------------------------------------------------------------------------------
streamfunction(sim::CavitySimulation) = streamfunction(sim.q, sim.ops)
vorticity(sim::CavitySimulation) = vorticity(sim.q, sim.ops)
velocity(sim::CavitySimulation) = velocity(sim.q, sim.ops)
grid(sim::CavitySimulation) = sim.ops.grid
