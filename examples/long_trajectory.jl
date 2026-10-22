# One long trajectory, appended snapshot by snapshot to a flat Float64 binary file.
#
#   OPENBLAS_NUM_THREADS=8 julia --project=. examples/long_trajectory.jl /path/to/output.bin
#
# Each record is one (N+1)×(N+1) column-major Float64 array of q, written every `save_every`
# time units from the start of the run: snapshot k is at t = k·save_every (the state at t = 0
# is the deterministic initial condition, `CavitySimulation(params).q`). A sidecar
# <output>.meta records the parameters and the time of the first snapshot. Only q is stored —
# ψ, ω, u and v follow from it (see read_trajectory.jl).
#
# Resumable: rerun the same command and it continues from the last snapshot in the file.
# (ARK3 is a one-step scheme, so a single q is a complete state — the trajectory file is its
# own checkpoint.) A crash can leave a partial record at the end; it is discarded on resume.

using HighReCavity, LinearAlgebra, Printf

const N            = 192
const Re           = 30_000
const dt           = 6.25e-3         # = 0.1/16, about 2× below the ARK3 stability limit
const tend         = 10_000.0
const save_every   = 0.1             # must be an integer multiple of dt
const report_every = 60.0            # seconds of wall time between progress lines
const path         = isempty(ARGS) ?
    error("usage: julia --project=. examples/long_trajectory.jl <output.bin>") : ARGS[1]

hms(s) = @sprintf("%d:%02d:%02d", floor(Int, s / 3600), floor(Int, (s % 3600) / 60), floor(Int, s % 60))
bar(frac, w = 20) = (n = clamp(round(Int, frac * w), 0, w); "[" * "="^n * " "^(w - n) * "]")

function main()
    BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", string(max(1, Sys.CPU_THREADS ÷ 2)))))
    stride = round(Int, save_every / dt)
    @assert stride * dt ≈ save_every

    sim = CavitySimulation(CavityParameters(N = N, Re = Re, dt = dt))
    rec = (N + 1)^2 * sizeof(Float64)
    nsaved = isfile(path) ? filesize(path) ÷ rec : 0

    if nsaved > 0                                # resume from the last complete snapshot
        filesize(path) == nsaved * rec || open(io -> truncate(io, nsaved * rec), path, "r+")
        open(path) do io
            seek(io, (nsaved - 1) * rec); read!(io, sim.q)
        end
        sim.q_prev .= sim.q
        sim.t = nsaved * save_every              # snapshot k is at t = k·save_every
        sim.step = round(Int, sim.t / dt)
        @printf("resuming at snapshot %d, t = %.2f of %.1f\n", nsaved, sim.t, tend)
    else
        write(path * ".meta",
              "N $N\nRe $Re\ndt $dt\nt_first $save_every\nsave_every $save_every\nprecision f64\n")
        @printf("starting from rest: N = %d, Re = %g, dt = %g, %d BLAS threads\n",
                N, Re, dt, BLAS.get_num_threads())
    end
    nsteps = (round(Int, (tend - sim.t) / dt) ÷ stride) * stride
    @printf("%d steps to t = %.1f, %d snapshots, %.1f GB\n",
            nsteps, tend, nsteps ÷ stride, (nsaved + nsteps ÷ stride) * rec / 2^30)
    flush(stdout)

    k, t0, tlast = nsaved, time(), time()
    open(path, "a") do io
        run!(sim, nsteps; every = stride, callback = function (s)
            write(io, s.q)
            k += 1
            if time() - tlast >= report_every || k == nsaved + nsteps ÷ stride
                flush(io)
                tlast = time()
                el = tlast - t0
                done = (k - nsaved) * stride
                frac = done / nsteps
                @printf("%s %5.1f%%  t = %8.1f/%.0f  snap %6d  %5.1f ms/step  elapsed %s  ETA %s  max|ω| %5.1f  %.1f GB\n",
                        bar(frac), 100frac, s.t, tend, k, 1e3 * el / done,
                        hms(el), hms(el * (nsteps - done) / max(done, 1)),
                        maximum(abs, s.ω), filesize(path) / 2^30)
                flush(stdout)
            end
        end)
    end
    @printf("done: t = %.2f, %d snapshots, %.2f GB, %s\n",
            sim.t, k, filesize(path) / 2^30, hms(time() - t0))
end

main()
