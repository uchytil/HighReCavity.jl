# The ORIGINAL dense-inverse solver, loaded verbatim (lines 8–257 of the reference script:
# grid, operators, laplacian/biharmonic/convection, flattened system, boundary rows) into a
# test-only module.  Ground truth for the equivalence tests; nothing here is edited.
module ReferenceDense

using LinearAlgebra

const REFERENCE_FILE = joinpath(@__DIR__, "..", "reference", "cavity-flow-spectral-ab2-cn.jl")
let lines = readlines(REFERENCE_FILE)
    @assert startswith(lines[8], "struct ChebyshevGrid") && lines[257] == "end"
    include_string(@__MODULE__, join(lines[8:257], "\n"), "cavity-flow-spectral-ab2-cn.jl[8:257]")
end

# The original `step!` (lines 275–290) with its two index globals passed explicitly;
# the arithmetic is unchanged.
function step!(q, q_prev, F, ops, grid, Δt, Re, walls_idxs_flat, top_idxs_flat)
    rhs = laplacian(q, ops) + 0.5 * Δt/Re * biharmonic(q, ops) - Δt * ( 3/2 * convection(q, ops) - 1/2 * convection(q_prev, ops) )
    rhs_vec = view(rhs, :)
    apply_bcs_rhs!(rhs_vec, walls_idxs_flat, top_idxs_flat, grid)
    q_prev .= q
    q[:] .= F * rhs_vec
    apply_bcs_rhs!(q[:], walls_idxs_flat, top_idxs_flat, grid) # Enforece BCs strictly
end

"""
    DenseSystem(N, α, Δt, Re_internal)

Everything the original script builds before stepping: grid, operators, A with boundary
rows replaced, A⁻¹, index sets, and the initial (q, q_prev) as the script initialises them.
`Re_internal` is the number the original script passed (`30000 / 2`).
"""
struct DenseSystem
    grid; ops; A; Ainv; walls; top; Δt; Re
end
function DenseSystem(N, α, Δt, Re_internal)
    grid = ChebyshevGrid((N, N), (α, α), Float64)
    ops = CavityOperators(Matrix{Float64}, grid)
    walls, top = get_boundary_indices(grid.Ns)
    A = build_system_matrix(ops, Δt, Re_internal)
    apply_bcs_system!(A, walls, top)
    return DenseSystem(grid, ops, A, inv(A), walls, top, Δt, Re_internal)
end
function initial_state(s::DenseSystem)
    q = zeros((s.grid.Ns[1]+1, s.grid.Ns[2]+1))
    q_prev = copy(q)
    apply_bcs_rhs!(view(q, :), s.walls, s.top, s.grid)
    return q, q_prev
end
step!(q, q_prev, s::DenseSystem) = step!(q, q_prev, s.Ainv, s.ops, s.grid, s.Δt, s.Re, s.walls, s.top)
"Streamfunction of the original representation, ψ = Wx q Wy (as the script's post-processing)."
ops_psi(s::DenseSystem, q) = s.ops.q.Wx * q * s.ops.q.Wy'
"Solve the original system for a given explicit RHS (boundary rows overwritten by the lid data)."
function solve(s::DenseSystem, rhs::AbstractMatrix)
    r = copy(rhs); apply_bcs_rhs!(view(r, :), s.walls, s.top, s.grid)
    return reshape(s.Ainv * vec(r), size(rhs))
end

end # module
