# Phase 10 — scaling study.   julia --project=bench bench/scaling.jl
#
# Dense inverse: measured (assembly + inv + GEMV) for N ≤ NDENSE_FULL; for larger N only
# the per-step GEMV is measured with a *same-size random matrix* (identical memory traffic)
# and the inverse's construction cost is not measured (it scales like N⁶: 71 s at N=128).
using HighReCavity, LinearAlgebra, Printf, Logging, BenchmarkTools

const Δt = 0.0005
const Re = 15000.0
const NDENSE_FULL = 128
const NDENSE_GEMV = 192          # (193²)² × 8 B = 11 GB; 256 would need 35 GB
Ns = [32, 64, 96, 128, 160, 192, 256]
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
ms(b) = minimum(b).time / 1e6
GB(x) = x / 2^30
MB(x) = x / 2^20

println("BLAS threads: ", BLAS.get_num_threads())
@printf("%5s | %10s %10s %10s | %10s %10s %10s %10s | %9s %9s | %8s %9s %9s\n",
        "N", "dense GEMV", "dense init", "dense mem", "inf A schur", "inf A ceig", "inf B schur", "inf B ceig",
        "inf init", "infB init", "inf mem", "infB mem", "RHS")
@printf("%5s | %10s %10s %10s | %10s %10s %10s %10s | %9s %9s | %8s %9s %9s\n",
        "", "[ms]", "[s]", "[GB]", "[ms]", "[ms]", "[ms]", "[ms]", "[s]", "[s]", "[MB]", "[MB]", "[ms]")
open("notes/scaling_results.txt", "w") do io
for N in Ns
    grid = ChebyshevGrid((N, N), (0.96, 0.96), Float64)
    ops = CavityOperators(Matrix{Float64}, grid)
    n = (N+1)^2
    # ---- dense ----
    dense_gemv = NaN; dense_init = NaN; dense_mem = GB(8.0 * n^2)
    if N <= NDENSE_FULL
        dense_init = @elapsed sys = ReferenceSystem(grid, ops, Δt, Re)
        rhsv = randn(n); out = similar(rhsv)
        dense_gemv = ms(@benchmark mul!($out, $(sys.Ainv), $rhsv))
        sys = nothing; GC.gc()
    elseif N <= NDENSE_GEMV
        M = randn(n, n); rhsv = randn(n); out = similar(rhsv)
        dense_gemv = ms(@benchmark mul!($out, $M, $rhsv))
        M = nothing; GC.gc()
    end
    # ---- influence ----
    t_A = @elapsed S_A = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false)
    end
    S_Ac = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false, poisson_mode = :ceigen)
    end
    t_B = @elapsed S_B = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false, build_response = true)
    end
    S_Bc = with_logger(NullLogger()) do
        InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false, build_response = true, poisson_mode = :ceigen)
    end
    st = QState(S_A)
    for i in 1:3; step_influence!(st, S_A); end
    rhs = copy(st.rhs); ω = zeros(N+1, N+1); qn = zeros(N+1, N+1)
    tA  = ms(@benchmark solve_timestep_linear!($ω, $qn, $rhs, $S_A; method = :resolve))
    tAc = ms(@benchmark solve_timestep_linear!($ω, $qn, $rhs, $S_Ac; method = :resolve))
    tB  = ms(@benchmark solve_timestep_linear!($ω, $qn, $rhs, $S_B; method = :response))
    tBc = ms(@benchmark solve_timestep_linear!($ω, $qn, $rhs, $S_Bc; method = :response))
    tR  = ms(@benchmark qform_rhs!($(st.rhs), $(st.q), $(st.q_prev), $ops, $Δt, $Re, $(st.W)))
    line = @sprintf("%5d | %10.3f %10.2f %10.3f | %10.3f %10.3f %10.3f %10.3f | %9.2f %9.2f | %8.1f %9.1f %9.3f",
            N, dense_gemv, dense_init, dense_mem, tA, tAc, tB, tBc, t_A, t_B,
            MB(Base.summarysize(S_A)), MB(Base.summarysize(S_B)), tR)
    println(line); println(io, line); flush(io)
    S_A = S_Ac = S_B = S_Bc = nothing; GC.gc()
end
end
