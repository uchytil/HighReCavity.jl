# One long trajectory, appended snapshot by snapshot to a flat Float64 binary file.
#
#   OPENBLAS_NUM_THREADS=8 julia --project=. examples/long_trajectory.jl /path/to/output.bin
#
# Each record is one (N+1)×(N+1) column-major Float64 array of q, written every `save_every`
# time units; snapshot k is at t = transient + k·save_every. A sidecar <output>.meta records
# the parameters. Only q is stored — ψ, ω, u and v follow from it (see read_trajectory.jl).
#
# Resumable: rerun the same command and it continues from the last snapshot in the file.
# (ARK3 is a one-step scheme, so a single q is a complete state — the trajectory file is its
# own checkpoint.) A crash can leave a partial record at the end; it is discarded on resume.

using HighReCavity, LinearAlgebra, Printf

const N          = 192
const Re         = 30_000
const dt         = 6.25e-3           # = 0.1/16, about 2× below the ARK3 stability limit
const transient  = 1_500.0           # advanced without saving
const tend       = 10_000.0
const save_every = 0.1               # must be an integer multiple of dt
const path       = isempty(ARGS) ?
    error("usage: julia --project=. examples/long_trajectory.jl <output.bin>") : ARGS[1]

function main()
    BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", string(max(1, Sys.CPU_THREADS ÷ 2)))))
    stride = round(Int, save_every / dt)
    @assert stride * dt ≈ save_every
    @assert (transient / dt) % stride == 0

    sim = CavitySimulation(CavityParameters(N = N, Re = Re, dt = dt))
    rec = (N + 1)^2 * sizeof(Float64)
    nsaved = isfile(path) ? filesize(path) ÷ rec : 0

    if nsaved > 0                                # resume from the last complete snapshot
        filesize(path) == nsaved * rec || open(io -> truncate(io, nsaved * rec), path, "r+")
        open(path) do io
            seek(io, (nsaved - 1) * rec); read!(io, sim.q)
        end
        sim.q_prev .= sim.q
        sim.t = transient + nsaved * save_every   # snapshot k is at transient + k·save_every
        sim.step = round(Int, sim.t / dt)
        @printf("resuming at snapshot %d, t = %.2f\n", nsaved, sim.t)
    else
        write(path * ".meta", "N $N\nRe $Re\ndt $dt\ntransient $transient\nsave_every $save_every\nprecision f64\n")
        @printf("transient: %d steps to t = %.1f\n", round(Int, transient / dt), transient)
        run!(sim, round(Int, transient / dt))
    end
    flush(stdout)

    nsteps = (round(Int, (tend - sim.t) / dt) ÷ stride) * stride
    k, t0 = nsaved, time()
    open(path, "a") do io
        run!(sim, nsteps; every = stride, callback = function (s)
            write(io, s.q)
            k += 1
            if k % 100 == 0
                flush(io)
                el = time() - t0
                @printf("t = %8.2f  snapshot %6d  %5.1f ms/step  %5.2f h elapsed  max|ω| %.1f\n",
                        s.t, k, 1e3 * el / ((k - nsaved) * stride), el / 3600, maximum(abs, s.ω))
                flush(stdout)
            end
        end)
    end
    @printf("done: t = %.2f, %d snapshots, %.2f GB, %.2f h\n",
            sim.t, k, filesize(path) / 2^30, (time() - t0) / 3600)
end

main()
