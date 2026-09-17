# Phase 8 — numerical comparison of the influence-matrix solver against the
# original dense-inverse solver.  Run:  julia --project=. scripts/compare_reference.jl [Nmax]
using HighReCavity, LinearAlgebra, Printf, Logging, Random

const Δt = 0.0005
const Re = 15000.0
Nmax = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
Ns = filter(<=(Nmax), [32, 64, 128])

relerr(a, b) = (n = norm(b); n == 0 ? norm(a - b) : norm(a - b) / n)
abserr(a, b) = maximum(abs.(a - b))
fmt(a, b) = @sprintf("abs %.2e  rel %.2e", abserr(a, b), relerr(a, b))

for N in Ns
    println("=" ^ 100)
    println("N = $N")
    grid = ChebyshevGrid((N, N), (0.96, 0.96), Float64)
    ops = CavityOperators(Matrix{Float64}, grid)
    tref = @elapsed sys = ReferenceSystem(grid, ops, Δt, Re; factorization = :lu)
    tinf = @elapsed S = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; build_response = true, verbose = false)
    end
    tpo = @elapsed SP = with_logger(NullLogger()) do
        InfluenceSolver(PsiOmegaForm, grid, ops, Δt, Re; verbose = false)
    end
    Sc = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false, poisson_mode = :ceigen)
    end
    L = S.layout
    @printf("init: reference inv(A) %.2f s | influence (QForm) %.2f s | influence (PsiOmega) %.2f s\n", tref, tinf, tpo)
    @printf("influence matrix QForm: m=%d rank=%d κ=%.2e | PsiOmega: rank=%d (m-4 expected) σ_max=%.2e\n",
            L.m, S.rank, S.svals[1]/S.svals[end], SP.rank, SP.svals[1])

    # --- bring the reference to a developed state (50 steps) ---
    q, qp = reference_initial_state(sys)
    for n in 1:50; step_reference!(q, qp, sys); end
    rhs = reference_rhs(q, qp, sys)

    # --- roundoff floor of the reference itself: inv(A)*rhs vs LU solve ---
    q_inv = reshape(sys.Ainv * vec(rhs), N+1, N+1)
    q_lu = reshape(sys.Alu \ vec(rhs), N+1, N+1)
    @printf("reference floor  inv(A)*rhs vs A\\rhs         : %s\n", fmt(q_inv, q_lu))

    # --- one linear solve, same RHS ---
    q_new = zeros(N+1, N+1); ω_new = zeros(N+1, N+1)
    solve_timestep_linear!(ω_new, q_new, rhs, S)
    q_newB = zeros(N+1, N+1); ω_newB = zeros(N+1, N+1)
    solve_timestep_linear!(ω_newB, q_newB, rhs, S; method = :response)
    @printf("one solve  q     (Method A vs inv(A))        : %s\n", fmt(q_new, q_inv))
    @printf("one solve  q     (Method B vs inv(A))        : %s\n", fmt(q_newB, q_inv))
    @printf("one solve  q     (Method A vs LU)            : %s\n", fmt(q_new, q_lu))
    q_newC = zeros(N+1, N+1); ω_newC = zeros(N+1, N+1)
    solve_timestep_linear!(ω_newC, q_newC, rhs, Sc)
    @printf("one solve  q     (Method A, ceigen vs inv(A)): %s\n", fmt(q_newC, q_inv))
    @printf("one solve  ω     (vs laplacian(q_inv))       : %s\n", fmt(ω_new, laplacian(q_inv, ops)))
    @printf("one solve  ω     (self-consistency Δψ)       : %s\n", fmt(ω_new, laplacian(q_new, ops)))
    Ψn = streamfunction_from_q(q_new, ops); Ψr = streamfunction_from_q(q_inv, ops)
    @printf("one solve  ψ                                 : %s\n", fmt(Ψn, Ψr))
    un, vn = velocity(q_new, ops); ur, vr = velocity(q_inv, ops)
    @printf("one solve  u                                 : %s\n", fmt(un, ur))
    @printf("one solve  v                                 : %s\n", fmt(vn, vr))
    @printf("one solve  convection                        : %s\n", fmt(convection(q_new, ops), convection(q_inv, ops)))
    @printf("one solve  q|Γ vs lid data                   : %.2e   ψ|Γ max: %.2e\n", abserr(gather(q_new, L), S.g), maximum(abs.(gather(Ψn, L))))
    h = lid_normal_derivative_target(grid, ops, L)
    @printf("one solve  ∂ₙψ (polynomial sense) − h        : %.2e   (reference: %.2e)\n",
            abserr(normal_derivative_boundary_q(q_new, ops, L), h), abserr(normal_derivative_boundary_q(q_inv, ops, L), h))
    dN = normal_derivative_boundary(Ψn, ops, L); dNr = normal_derivative_boundary(Ψr, ops, L)
    @printf("one solve  ∂ₙψ (P_N nodal sense) new vs ref  : %s   | nodal ∂ₙψ − h (ref): %.2e\n", fmt(dN, dNr), abserr(dNr, h))

    # --- PsiOmegaForm from the same RHS: a different discretization ---
    ψP = zeros(N+1, N+1); ωP = zeros(N+1, N+1)
    solve_timestep_linear!(ωP, ψP, rhs, SP)
    @printf("PsiOmega   ψ vs reference ψ                  : %s\n", fmt(ψP, Ψr))
    @printf("PsiOmega   ω vs reference ω (interior)       : %s\n", fmt(ωP[2:N, 2:N], laplacian(q_inv, ops)[2:N, 2:N]))
    @printf("PsiOmega   ω vs reference ω (wall)           : %s\n", fmt(gather(ωP, L), gather(laplacian(q_inv, ops), L)))
    rP = normal_derivative_boundary(ψP, ops, L) - h
    @printf("PsiOmega   ∂ₙψ − h  (LS residual, corners)   : %.2e  (in left null space: %.2e, outside: %.2e)\n",
            maximum(abs.(rP)), norm(SP.Cfact.U' * rP) == 0 ? 0.0 : norm(rP - SP.Cfact.U * (SP.Cfact.U' * rP)), norm(SP.Cfact.U' * rP))

    # --- trajectories: 200 more steps from the same state ---
    st = QState(S); st.q .= q; st.q_prev .= qp
    stB = QState(S); stB.q .= q; stB.q_prev .= qp
    stC = QState(Sc); stC.q .= q; stC.q_prev .= qp
    nsteps = 200
    tr = @elapsed for n in 1:nsteps; step_reference!(q, qp, sys); end
    ta = @elapsed for n in 1:nsteps; step_influence!(st, S; method = :resolve); end
    tb = @elapsed for n in 1:nsteps; step_influence!(stB, S; method = :response); end
    tc = @elapsed for n in 1:nsteps; step_influence!(stC, Sc; method = :resolve); end
    @printf("%d steps  wall: reference %.3f s | influence A %.3f s | influence B %.3f s\n", nsteps, tr, ta, tb)
    @printf("%d steps  q   (A)                            : %s\n", nsteps, fmt(st.q, q))
    @printf("%d steps  q   (B)                            : %s\n", nsteps, fmt(stB.q, q))
    @printf("%d steps  q   (A, ceigen)  [%.3f s]          : %s\n", nsteps, tc, fmt(stC.q, q))
    @printf("%d steps  ψ   (A)                            : %s\n", nsteps, fmt(streamfunction_from_q(st.q, ops), streamfunction_from_q(q, ops)))
    @printf("%d steps  ω   (A)                            : %s\n", nsteps, fmt(st.ω, laplacian(q, ops)))
    ua, va = velocity(st.q, ops); ur, vr = velocity(q, ops)
    @printf("%d steps  u   (A)                            : %s\n", nsteps, fmt(ua, ur))
    @printf("%d steps  v   (A)                            : %s\n", nsteps, fmt(va, vr))
    @printf("%d steps  N   (A)                            : %s\n", nsteps, fmt(convection(st.q, ops), convection(q, ops)))
    @printf("%d steps  max|q| = %.3e, max|ω| = %.3e\n", nsteps, maximum(abs.(q)), maximum(abs.(laplacian(q, ops))))
end
