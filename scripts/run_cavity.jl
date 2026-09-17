# Lid-driven cavity with the influence-matrix solver — same physical set-up as the
# reference script (N = 128, α = 0.96, Δt = 5e-4, Re = 15000, regularised lid u = (1−x²)²).
#
#   julia --project=. scripts/run_cavity.jl [N] [num_steps] [poisson_mode]
#
# poisson_mode = schur (default, most accurate) | ceigen (≈4× faster at N=128, see the note)
# The final state is serialised to results/cavity_N<N>_steps<n>.jls  (q, q_prev, ω, grid parameters).
using HighReCavity, LinearAlgebra, Printf, Serialization

N         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
num_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 60000
pmode     = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :schur
Δt = 0.0005
Re = 30000 / 2.0

grid = ChebyshevGrid((N, N), (0.96, 0.96), Float64)
ops = CavityOperators(Matrix{Float64}, grid)
tinit = @elapsed solver = InfluenceSolver(QForm, grid, ops, Δt, Re; poisson_mode = pmode)
@printf("initialised influence solver (N=%d, poisson_mode=%s) in %.2f s\n", N, pmode, tinit)

st = QState(solver)              # q = lid data on the boundary, zero interior; q_prev = 0 (as in the reference)
t0 = time()
for n in 1:num_steps
    step_influence!(st, solver)
    if n % 1000 == 0 || n == num_steps
        @printf("step %6d / %d   max|ω| = %.4e   (%.1f ms/step)\n", n, num_steps, maximum(abs, st.ω), 1e3 * (time() - t0) / n)
    end
    any(isnan, st.q) && error("Solution diverged at step $n")
end

mkpath("results")
out = joinpath("results", "cavity_N$(N)_steps$(num_steps).jls")
serialize(out, (q = st.q, q_prev = st.q_prev, ω = st.ω, N = N, α = 0.96, Δt = Δt, Re = Re, num_steps = num_steps))
println("saved ", out)

# Post-processing (velocity, vorticity, streamfunction) exactly as in the reference:
u, v = velocity(st.q, ops)
ψ = streamfunction_from_q(st.q, ops)
@printf("max|u| = %.4f, max|v| = %.4f, min ψ = %.5f, max ψ = %.5f\n", maximum(abs, u), maximum(abs, v), minimum(ψ), maximum(ψ))
# For plots use the reference's `interp_matrices` on a uniform grid, e.g.
#   xs = range(-1, 1, length = 200); Ix, Iy = interp_matrices((collect(xs), collect(xs)), grid); Ix * ψ * Iy'
