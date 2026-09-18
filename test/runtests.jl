using HighReCavity
using Test, LinearAlgebra, Random

include("reference_dense.jl")
using .ReferenceDense: DenseSystem

const dt = 0.0005
const Re_full = 30_000.0          # public convention (full side length L = 2)
const Re_old  = 15_000.0          # what the original script used internally (30000 / 2)
const α = 0.96

relerr(a, b) = norm(a - b) / norm(b)

"Regression data written by the preserved validated implementation (tag v0.1.0-validated)."
function read_validated(path)
    open(path) do io
        out = Matrix{Float64}[]
        while !eof(io)
            dims = parse.(Int, split(readline(io)))
            push!(out, reshape([parse(Float64, readline(io)) for _ in 1:prod(dims)], dims...))
        end
        out
    end
end

@testset "HighReCavity" begin

@testset "1. small-N dense equivalence (N = 8, 12)" begin
    Random.seed!(1)
    for N in (8, 12), backend in (:ceigen, :schur)
        ref = DenseSystem(N, α, dt, Re_old)
        sim = CavitySimulation(CavityParameters(N = N, Re = Re_full, dt = dt, alpha = α, backend = backend))
        for trial in 1:3
            f = randn(N+1, N+1)
            q_ref = ReferenceDense.solve(ref, f)
            q = zeros(N+1, N+1); ω = zeros(N+1, N+1)
            influence_solve!(q, ω, f, sim.solver)
            @test relerr(q, q_ref) < 1e-12
            @test relerr(ω, ReferenceDense.laplacian(q_ref, ref.ops)) < 1e-11      # ω = L q
        end
        # complete affine map rhs ↦ q, column by column
        q0 = zeros(N+1, N+1); ω = zeros(N+1, N+1)
        influence_solve!(q0, ω, zeros(N+1, N+1), sim.solver)
        li = LinearIndices((N+1, N+1)); int = vec(li[2:N, 2:N])
        for idx in int
            e = zeros(N+1, N+1); e[idx] = 1
            q = zeros(N+1, N+1); influence_solve!(q, ω, e, sim.solver)
            @test maximum(abs.((q - q0) - reshape(ref.Ainv[:, idx], N+1, N+1))) < 1e-13
        end
    end
end

@testset "2. short trajectory vs validated implementation" begin
    for (N, nsteps) in ((8, 3), (24, 100))
        q_v, qp_v, ω_v = read_validated(joinpath(@__DIR__, "data", "validated_N$(N)_steps$(nsteps).txt"))
        for backend in (:ceigen, :schur)
            sim = CavitySimulation(CavityParameters(N = N, Re = Re_full, dt = dt, alpha = α, backend = backend))
            run!(sim, nsteps)
            @test sim.step == nsteps && sim.t ≈ nsteps * dt
            @test relerr(sim.q, q_v) < 1e-10
            @test relerr(sim.q_prev, qp_v) < 1e-10
            @test relerr(sim.ω, ω_v) < 1e-10
        end
        # and against the original dense solver stepped here
        ref = DenseSystem(N, α, dt, Re_old)
        q, qp = ReferenceDense.initial_state(ref)
        sim = CavitySimulation(CavityParameters(N = N, Re = Re_full, dt = dt, alpha = α))
        @test sim.q == q && sim.q_prev == qp
        for n in 1:nsteps; ReferenceDense.step!(q, qp, ref); end
        run!(sim, nsteps)
        @test relerr(sim.q, q) < 1e-10
        @test relerr(streamfunction(sim), ReferenceDense.ops_psi(ref, q)) < 1e-10
        u, v = velocity(sim); u_r, v_r = ReferenceDense.velocity(q, ref.ops)
        @test relerr(u, u_r) < 1e-10 && relerr(v, v_r) < 1e-10
        @test relerr(vorticity(sim), ReferenceDense.vorticity(q, ref.ops)) < 1e-10
    end
end

@testset "3. boundary conditions" begin
    N = 16
    sim = CavitySimulation(CavityParameters(N = N, Re = Re_full, dt = dt, alpha = α))
    x = grid(sim).x
    L = sim.solver.layout
    for n in 1:5
        step!(sim)
        q = sim.q
        @test q[:, end] == -1/2 .* (1 .- x.^2)                           # lid: q = −½(1−x²)
        @test all(q[:, 1] .== 0) && all(q[1, :] .== 0) && all(q[end, :] .== 0)
        ψ = streamfunction(sim)
        @test all(ψ[[1, end], :] .== 0) && all(ψ[:, [1, end]] .== 0)      # ψ|Γ = 0
        u, v = velocity(sim)
        @test u[:, end] ≈ (1 .- x.^2).^2 atol = 1e-14                     # lid velocity (1−x²)²
        @test maximum(abs.(u[:, 1])) < 1e-14 && maximum(abs.(v[[1, end], :])) < 1e-14   # no slip
        @test maximum(abs.(v[:, [1, end]])) < 1e-14 && maximum(abs.(u[[1, end], :])) < 1e-14  # no penetration
        ω = vorticity(sim)
        @test relerr(sim.ω, ω) < 1e-13                                   # wall-vorticity closure ω = L q
        @test all(sim.ω[[1, end], [1, end]] .== 0)
    end
end

@testset "4. backend agreement (:ceigen vs :schur)" begin
    for N in (16, 32)
        pc = CavityParameters(N = N, Re = Re_full, dt = dt, alpha = α, backend = :ceigen)
        ps = CavityParameters(N = N, Re = Re_full, dt = dt, alpha = α, backend = :schur)
        sc = CavitySimulation(pc); ss = CavitySimulation(ps)
        @test sc.solver.M ≈ ss.solver.M rtol = 1e-10
        run!(sc, 20); run!(ss, 20)
        @test relerr(sc.q, ss.q) < 1e-11
        @test relerr(sc.ω, ss.ω) < 1e-11
    end
end

@testset "5. Reynolds-number convention (public Re = full side length)" begin
    p = CavityParameters(N = 16, Re = 30_000, dt = dt)
    @test p.Re == 30_000
    @test reynolds_internal(p) == 15_000
    @test HighReCavity.diffusion_coefficient(p) == 0.5 * dt / 15_000
    # one step with public Re = 30_000 equals the original run with Re = 15_000 …
    sim = CavitySimulation(CavityParameters(N = 16, Re = 30_000, dt = dt, alpha = α))
    ref = DenseSystem(16, α, dt, 15_000.0)
    q, qp = ReferenceDense.initial_state(ref)
    for n in 1:5; ReferenceDense.step!(q, qp, ref); step!(sim); end
    @test relerr(sim.q, q) < 1e-11
    # … and differs from the original run with Re = 30_000 (i.e. the conversion is real)
    ref2 = DenseSystem(16, α, dt, 30_000.0)
    q2, qp2 = ReferenceDense.initial_state(ref2)
    for n in 1:5; ReferenceDense.step!(q2, qp2, ref2); end
    @test relerr(sim.q, q2) > 1e-6
end

@testset "run! callback" begin
    sim = CavitySimulation(CavityParameters(N = 8, Re = Re_full, dt = dt))
    seen = Int[]
    run!(sim, 10; callback = s -> push!(seen, s.step), every = 4)
    @test seen == [4, 8, 10]
    @test_throws ArgumentError CavityParameters(N = 8, Re = 1.0, dt = dt, backend = :eigen)
end

@testset "6. ARK3 integrator" begin
    # temporal order of convergence on a resolved case (Re = 1000, N = 24): ARK3 third order,
    # CNAB2 second order; both integrators satisfy the wall data and ω = L q.
    N, Re = 24, 1000.0
    s0 = CavitySimulation(CavityParameters(N = N, Re = Re, dt = 5e-4)); run!(s0, 200); q0 = copy(s0.q)
    T = 0.16
    function advance(integ, dt)
        sim = CavitySimulation(CavityParameters(N = N, Re = Re, dt = dt, integrator = integ))
        sim.q .= q0; sim.q_prev .= q0
        run!(sim, round(Int, T / dt)); return sim
    end
    ref = advance(:ark3, 2.5e-4)
    e_ark = [relerr(advance(:ark3, dt).q, ref.q) for dt in (8e-3, 4e-3, 2e-3)]
    @test 6 < e_ark[1] / e_ark[2] < 10 && 6 < e_ark[2] / e_ark[3] < 10
    # ARK3 tableau: third-order conditions (shared weights b, explicit and implicit parts, coupling)
    γ = HighReCavity.ARK3_γ; b = collect(HighReCavity.ARK3_b); bhat = collect(HighReCavity.ARK3_bhat)
    AE = zeros(4, 4); AI = zeros(4, 4)
    for i in 2:4, j in 1:i-1; AE[i, j] = HighReCavity.ARK3_AE[i-1][j]; AI[i, j] = HighReCavity.ARK3_AI[i-1][j]; end
    for i in 2:4; AI[i, i] = γ; end
    c = vec(sum(AE, dims = 2))
    @test vec(sum(AI, dims = 2)) ≈ c
    @test AI[4, :] ≈ b                                   # stiffly accurate implicit part
    @test sum(b) ≈ 1 && b'c ≈ 1/2 && b' * (c.^2) ≈ 1/3
    @test b' * (AE * c) ≈ 1/6 && b' * (AI * c) ≈ 1/6
    @test sum(bhat) ≈ 1 && bhat'c ≈ 1/2
    # boundary conditions and consistency after ARK3 steps
    sim = CavitySimulation(CavityParameters(N = 16, Re = Re_full, dt = dt, alpha = α, integrator = :ark3))
    run!(sim, 5); x = grid(sim).x
    @test sim.q[:, end] == -1/2 .* (1 .- x.^2) && all(sim.q[:, 1] .== 0) && all(sim.q[[1, end], :] .== 0)
    @test relerr(sim.ω, vorticity(sim)) < 1e-13
    @test_throws ArgumentError CavityParameters(N = 8, Re = 1.0, dt = dt, integrator = :rk4)
end

end
