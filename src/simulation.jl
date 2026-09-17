# Simulation: parameters, state, one time step, long runs.
#
# Vorticity equation  ∂ₜω + u ω_x + v ω_y = ν Δω  with ω = Δψ, discretised in time by
# Crank–Nicolson for diffusion and second-order Adams–Bashforth for convection:
#
#     (I − cΔ) ω^{n+1} = f,    f = ω^n + c Δω^n − dt (3/2 N^n − 1/2 N^{n−1}),    c = ν dt/2,
#
# where N = u ω_x + v ω_y.  On the right-hand side ω^n = Δψ^n and Δω^n = Δ²ψ^n are formed from
# q^n with the product-rule operators (`laplacian!`, `biharmonic!`); on the left the Laplacian
# acts on the nodal vorticity (D2·Ω + Ω·D2ᵀ).  The implicit step is solved by
# `influence_solve!`.  On the first step N^{n−1} = 0 (q^{−1} = 0).

"""
    CavityParameters(; N, Re, dt, alpha = 0.96, backend = :ceigen, T = Float64)

Simulation parameters.

- `N`: polynomial degree; the grid has (N+1)² nodes.
- `Re`: Reynolds number U L / ν based on the full cavity side length L = 2 and the peak lid
  speed U = 1.
- `dt`: time step of the Crank–Nicolson / Adams–Bashforth scheme.
- `alpha`: grid mapping parameter, 0 ≤ alpha < 1 (0 = unmapped Chebyshev grid; see `ChebyshevGrid`).
- `backend`: solver for the q-Poisson stage, `:ceigen` (default) or `:schur` (see `SylvesterSolver`).
- `T`: floating-point type of the computation.
"""
struct CavityParameters{T<:AbstractFloat}
    N::Int
    Re::T             # based on the full side length L = 2
    dt::T
    alpha::T
    backend::Symbol   # q-Poisson stage: :ceigen or :schur
end

function CavityParameters(; N::Int, Re, dt, alpha = 0.96, backend::Symbol = :ceigen, T::Type{<:AbstractFloat} = Float64)
    backend in (:ceigen, :schur) || throw(ArgumentError("backend must be :ceigen or :schur"))
    return CavityParameters{T}(N, T(Re), T(dt), T(alpha), backend)
end

"""
    reynolds_internal(params)

Reynolds number in the units of the discrete operators.  The operators use the cavity
half-width as length scale while `params.Re` uses the full side length L = 2, so this is
`Re / 2` (kinematic viscosity ν = 2/Re).
"""
reynolds_internal(p::CavityParameters) = p.Re / 2

"Diffusion coefficient of the implicit step, c = ν dt / 2 = dt / (2 Re_internal)."
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
    ω::Matrix{T}          # vorticity Δψⁿ, produced by the last implicit solve (zero before the first step)
    f::Matrix{T}          # right-hand side work array
    work::Workspace{T}
    step::Int             # number of steps taken
    t::T                  # current time
end

"""
    CavitySimulation(params::CavityParameters)

Sets up a simulation: grid, operators, Sylvester decompositions, influence matrix and work
arrays (all fixed cost).  The initial state is fluid at rest with the lid impulsively started:
q = 0 in the interior, lid data on the walls.  Fields: `q`, `q_prev`, `ω`, `step`, `t`,
`params`, `ops`.
"""
function CavitySimulation(p::CavityParameters{T}) where {T}
    grid = ChebyshevGrid(p.N, p.alpha)
    ops = CavityOperators(grid)
    solver = InfluenceSolver(ops, diffusion_coefficient(p); backend = p.backend)
    n = p.N + 1
    q = zeros(T, n, n)
    q_prev = copy(q)                                   # q⁻¹ = 0, so N⁻¹ = 0 on the first step
    set_walls!(q, solver.g, solver.layout)
    return CavitySimulation{T,typeof(solver)}(p, ops, solver, q, q_prev, zeros(T, n, n), zeros(T, n, n), Workspace{T}(n), 0, zero(T))
end

"""
    rhs!(f, q, q_prev, ops, dt, c, W)  —  f = ωⁿ + c Δωⁿ − dt (3/2 Nⁿ − 1/2 Nⁿ⁻¹)

Explicit part of the time step, with ωⁿ = Δψⁿ and Δωⁿ = Δ²ψⁿ formed from q by the
product-rule operators.
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
    step!(sim)

Advance one time step:  (qⁿ, qⁿ⁻¹) → right-hand side f → implicit solve → qⁿ⁺¹.
Updates `sim.q`, `sim.q_prev`, `sim.ω`, `sim.step` and `sim.t`.
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

Advance `nsteps` steps.  Only the current state is kept; `callback(sim)` is called after
every `every`-th step and after the last one, and is the place to save, inspect or process
the state during a long run.  With `check_nan` the run stops with an error if the solution
becomes NaN.
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
# Physical fields of the current state
# ---------------------------------------------------------------------------------------
streamfunction(sim::CavitySimulation) = streamfunction(sim.q, sim.ops)
vorticity(sim::CavitySimulation) = vorticity(sim.q, sim.ops)
velocity(sim::CavitySimulation) = velocity(sim.q, sim.ops)

"""
    grid(sim) -> ChebyshevGrid

The one-dimensional grid of the simulation; `grid(sim).x` are the node coordinates, used in
both directions.
"""
grid(sim::CavitySimulation) = sim.ops.grid
