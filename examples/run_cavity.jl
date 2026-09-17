# Long run: transient, then a data-collection interval with periodic processing.
#   julia --project=. examples/run_cavity.jl
using HighReCavity, Printf, Serialization

params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4, alpha = 0.96, backend = :ceigen)
sim = CavitySimulation(params)

# 1. transient (nothing stored; progress every 1000 steps)
run!(sim, 20_000; every = 1000, callback = s -> @printf("t = %7.2f  max|ω| = %.3e\n", s.t, maximum(abs, s.ω)))

# 2. production: process the state every 200 steps — here, accumulate a time series and
#    snapshots; replace with whatever the post-processing needs.
mkpath("results")
energy = Float64[]
function collect!(s)
    u, v = velocity(s)
    push!(energy, sum(abs2, u) + sum(abs2, v))
    s.step % 5000 == 0 && serialize(@sprintf("results/q_step%07d.jls", s.step), (q = s.q, q_prev = s.q_prev, t = s.t, params = s.params))
end
run!(sim, 40_000; every = 200, callback = collect!)
serialize("results/energy.jls", energy)

ψ = streamfunction(sim); ω = vorticity(sim)
@printf("done: t = %.2f, min ψ = %.5f, max|ω| = %.3e\n", sim.t, minimum(ψ), maximum(abs, ω))
