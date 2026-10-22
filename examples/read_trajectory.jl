# Read a trajectory written by examples/long_trajectory.jl, without loading it into memory.
#
#   include("examples/read_trajectory.jl")
#   Q, t = trajectory("output.bin")          # Q is memory-mapped (n, n, nsnap)
#   q = Q[:, :, 500];  t[500]
#
# Derived fields come from the package operators, e.g.
#   ops = CavityOperators(ChebyshevGrid(size(Q, 1) - 1, 0.96))
#   ω = vorticity(Q[:, :, 500], ops);  u, v = velocity(Q[:, :, 500], ops)

using Mmap

function trajectory(path::AbstractString)
    meta = Dict(split(l)[1] => split(l)[2] for l in readlines(path * ".meta"))
    n = parse(Int, meta["N"]) + 1
    t0 = parse(Float64, meta["transient"]); Δ = parse(Float64, meta["save_every"])
    nsnap = filesize(path) ÷ (n^2 * sizeof(Float64))
    Q = open(io -> Mmap.mmap(io, Array{Float64,3}, (n, n, nsnap)), path)
    return Q, t0 .+ (1:nsnap) .* Δ        # the first snapshot is one interval after the transient
end
