# Phase 9 — per-timestep benchmark of the linear solve variants.
#   julia --project=bench bench/benchmark.jl [Nmax]
using HighReCavity, LinearAlgebra, Printf, Logging, BenchmarkTools

const Δt = 0.0005
const Re = 15000.0
Nmax = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
Ns = filter(<=(Nmax), [32, 64, 128])
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 200

@printf("BLAS threads: %d  (%s)\n", BLAS.get_num_threads(), BLAS.get_config().loaded_libs[1].libname)
GB(x) = x / 2^30
MB(x) = x / 2^20
tstr(b) = @sprintf("%9.3f ms  (min %8.3f ms, allocs %6d, %8.1f KiB)", median(b).time/1e6, minimum(b).time/1e6, b.allocs, b.memory/1024)

for N in Ns
    println("=" ^ 100)
    println("N = $N   ((N+1)² = $((N+1)^2) unknowns)")
    grid = ChebyshevGrid((N, N), (0.96, 0.96), Float64)
    ops = CavityOperators(Matrix{Float64}, grid)

    # --- initialisation cost & memory ---
    t_inv = @elapsed sys = ReferenceSystem(grid, ops, Δt, Re)
    t_lu = @elapsed Alu = lu(sys.A)
    t_A = @elapsed S_A = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false)
    end
    t_B = @elapsed S_B = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false, build_response = true)
    end
    @printf("init   inv(A) (incl. assembly): %8.2f s   memory Ainv %.3f GB (+A %.3f GB)\n", t_inv, GB(sizeof(sys.Ainv)), GB(sizeof(sys.A)))
    @printf("init   lu(A):                   %8.2f s   memory LU   %.3f GB\n", t_lu, GB(sizeof(Alu.factors)))
    @printf("init   influence (Method A):    %8.2f s   memory      %.1f MB\n", t_A, MB(Base.summarysize(S_A)))
    @printf("init   influence (Method B):    %8.2f s   memory      %.1f MB  (Rω, Rq: %.1f MB)\n", t_B, MB(Base.summarysize(S_B)), MB(sizeof(S_B.Rω) + sizeof(S_B.RX)))

    # --- a physical state / rhs ---
    q, qp = reference_initial_state(sys)
    for n in 1:20; step_reference!(q, qp, sys); end
    rhs = reference_rhs(q, qp, sys)
    rhsv = vec(rhs); out = similar(rhsv)
    st = QState(S_A); st.q .= q; st.q_prev .= qp
    ω = zeros(N+1, N+1); qn = zeros(N+1, N+1)

    b_inv = @benchmark mul!($out, $(sys.Ainv), $rhsv)
    b_lu  = @benchmark ldiv!($out, $Alu, $rhsv)
    b_A   = @benchmark solve_timestep_linear!($ω, $qn, $rhs, $S_A; method = :resolve)
    b_B   = @benchmark solve_timestep_linear!($ω, $qn, $rhs, $S_B; method = :response)
    b_rhs = @benchmark qform_rhs!($(st.rhs), $(st.q), $(st.q_prev), $ops, $Δt, $Re, $(st.W))
    b_ref = @benchmark step_reference!($(copy(q)), $(copy(qp)), $sys)
    b_stA = @benchmark step_influence!($st, $S_A; method = :resolve)
    b_stB = @benchmark step_influence!($st, $S_B; method = :response)
    println("per-step linear solve:")
    @printf("  1. Ainv * rhs (GEMV)                 %s\n", tstr(b_inv))
    @printf("  2. LU triangular solves              %s\n", tstr(b_lu))
    @printf("  3. influence, fresh solves (A)       %s\n", tstr(b_A))
    @printf("  4. influence, response GEMV (B)      %s\n", tstr(b_B))
    println("other:")
    @printf("  explicit RHS (in-place)              %s\n", tstr(b_rhs))
    @printf("  full reference step (alloc. RHS+inv) %s\n", tstr(b_ref))
    @printf("  full influence step, Method A        %s\n", tstr(b_stA))
    @printf("  full influence step, Method B        %s\n", tstr(b_stB))
    # breakdown of Method A
    b_h = @benchmark solve_helmholtz_dirichlet!($ω, $rhs, $(S_A.helm))
    b_p = @benchmark solve_qpoisson_dirichlet!($qn, $ω, $(S_A.pois))
    b_c = @benchmark HighReCavity.closure!($(S_A.d0), $qn, $ω, $S_A)
    b_i = @benchmark HighReCavity._solve_influence!($(S_A.ξ), $S_A, $(S_A.d0))
    println("Method A breakdown (each done twice per step):")
    @printf("  Helmholtz (eigen, 4 GEMM)            %s\n", tstr(b_h))
    @printf("  q-Poisson (Schur, 4 GEMM + trsyl)    %s\n", tstr(b_p))
    @printf("  closure  (wall laplacian)            %s\n", tstr(b_c))
    @printf("  influence LU solve (m=%d)            %s\n", S_A.layout.m, tstr(b_i))
    sys = nothing; Alu = nothing; GC.gc()
end
