using LinearAlgebra
using CairoMakie
using StaticArrays
using SparseArrays
using LinearMaps
using Krylov

struct ChebyshevGrid{T<:AbstractFloat, D}
    Ns::NTuple{D, Int} # Number of points in each dimension
    ηs::NTuple{D, Vector{T}} # Logical nodes (standard Chebyshev)
    xs::NTuple{D, Vector{T}} # Physical nodes (mapped)
    αs::NTuple{D, T} # Mapping parameter for each dimension
end

function ChebyshevGrid(Ns::NTuple{D, Int}, αs::NTuple{D, T}=ntuple(i->0.96, D), ::Type{T}=Float64) where {D, T<:AbstractFloat}
    ηs = ntuple(d ->  -cos.(T(π) .* T.(0:Ns[d]) ./ T(Ns[d])), D)

    xs = ntuple(D) do d
        if αs[d] > 0
            L = asin(αs[d])
            asin.(αs[d] .* ηs[d]) ./ L
        else
            ηs[d]
        end
    end

    return ChebyshevGrid{T, D}(Ns, ηs, xs, αs)
end


function diff_matrix(::Type{M}, η::Vector{T}, α::T) where {T<:AbstractFloat, M<:Matrix}
    N = length(η) - 1
    
    # 1. Build standard Chebyshev D using logical η
    c = [2; ones(N-1); 2] .* T.((-1).^(0:N))
    X = repeat(η, 1, N+1)
    dX = X - X'
    D_std = (c * (1 ./ c)') ./ (dX .+ I(N+1))
    D_std = D_std - diagm(vec(sum(D_std, dims=2)))
    
    # 2. Calculate the Jacobian directly from η
    if α > 0
        L = asin(α)
        # Jacobian J = 1 / (dy/dη)
        # dy/dη = α / (L * sqrt(1 - (α*η)^2))
        inv_metric = @. (L * sqrt(1 - (α * η)^2)) / α
        D1 = inv_metric .* D_std
    else
        D1 = D_std
    end
    
    # 3. Higher orders via nesting (matrix multiplication)
    D2 = D1 * D1
    
    return D1, D2
end

function diff_matrices(::Type{M}, grid::ChebyshevGrid{T, D}) where {T<:AbstractFloat, D, M <: AbstractMatrix}
    ntuple(d -> diff_matrix(M, grid.ηs[d], grid.αs[d]), D)
end

function weights(N::Int, ::Type{T}=Float64) where T<:AbstractFloat
    w = ones(T, N+1)
    w[1] = T(0.5)
    w[end] = T(0.5)
    w .*= T.((-1) .^(0:N))
    return w
end

function weights(grid::ChebyshevGrid{T, D}) where {T<:AbstractFloat, D}
    ntuple(d -> weights(grid.Ns[d]), D)
end

function interp_matrix(x_target::Vector{T}, η_source::Vector{T}, weights::Vector{T}, α::T) where T<:AbstractFloat
    N_t = length(x_target)
    N_s = length(η_source)
    M = zeros(N_t, N_s)

    η_target = nothing

    if α > 0
        η_target = sin.(asin(α) .* x_target) ./ α
    else
        η_target = x_target
    end
   
    for i in 1:N_t
        dη = η_target[i] .- η_source
        if any(dη .== 0)
            idx = findfirst(dη .== 0)
            M[i, idx] = 1.0
        else
            w = weights ./ dη
            M[i, :] = w ./ sum(w)
        end
    end
    return M
end

function interp_matrices(x_targets::NTuple{D, AbstractVector{T}}, grid::ChebyshevGrid{T, D}) where {T<:AbstractFloat, D}
    ws= weights(grid)
    ntuple(d -> interp_matrix(x_targets[d], grid.ηs[d], ws[d], grid.αs[d]), D)
end

abstract type ActsOn end  # marker for function space

struct Q <: ActsOn end
struct Psi <: ActsOn end

struct Operators{F <: ActsOn, T <: AbstractFloat, M <: AbstractMatrix{T}}
    Dx::M
    Dy::M
    D²x::M
    D²y::M
    D⁴x::M
    D⁴y::M
    Wx::Union{Diagonal{T, Vector{T}}, Nothing}
    Wy::Union{Diagonal{T, Vector{T}}, Nothing}
end

struct CavityOperators{T <: AbstractFloat, M <: AbstractMatrix{T}}
    q::Operators{Q, T, M}
    ψ::Operators{Psi, T, M}
end

function CavityOperators(::Type{M}, grid::ChebyshevGrid{T, 2}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    Nx, Ny = grid.Ns
    Nx, Ny = grid.Ns
    xs, ys = grid.xs
    (Dx, D2x), (Dy, D2y) = diff_matrices(M, grid)

    Wx = Diagonal(1 .- xs.^2)
    Wy = Diagonal(1 .- ys.^2)

    Dx_q = Wx * Dx - 2 * Diagonal(xs)
    Dy_q = Wy * Dy - 2 * Diagonal(ys)

    D²x_q = Wx * D2x - 4 * Diagonal(xs) * Dx - 2 * I(Nx+1) # Operator  ∂²/∂x² ψ  = Lx_q * q
    D²y_q = Wy * D2y - 4 * Diagonal(ys) * Dy - 2 * I(Ny+1) # Operator  ∂²/∂y² ψ = Ly_q * q

    D⁴x_q = Wx * D2x^2 - 8*Diagonal(xs) * D2x * Dx - 12*D2x
    D⁴y_q = Wy * D2y^2 - 8*Diagonal(ys) * D2y * Dy - 12*D2y


    q_ops = Operators{Q, T, M}(Dx_q, Dy_q, D²x_q, D²y_q, D⁴x_q, D⁴y_q, Wx, Wy)
    ψ_ops = Operators{Psi, T, M}(Dx, Dy, D2x, D2y, D2x^2, D2y^2, nothing, nothing)

    return CavityOperators{T,  M}(q_ops, ψ_ops)

end


function convection(q::Matrix{T}, ops::CavityOperators{T, M}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}

    u = ops.q.Wx * q * ops.q.Dy' 
    v = -ops.q.Dx * q * ops.q.Wy'
    ω = ops.q.Wx * q * ops.q.D²y' + ops.q.D²x * q * ops.q.Wy'
    ω_x = ops.ψ.Dx * ω
    ω_y = ω * ops.ψ.Dy'


    return u .*  ω_x + v .* ω_y
end

function laplacian(q::Matrix{T}, ops::CavityOperators{T, M}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    return ops.q.Wx * q * ops.q.D²y' + ops.q.D²x * q * ops.q.Wy'
end

function biharmonic(q::Matrix{T}, ops::CavityOperators{T, M}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    term1 = ops.q.Wx * q * ops.q.D⁴y'
    term2 = ops.q.D⁴x * q * ops.q.Wy'
    term3 = ops.ψ.D²x * (ops.q.Wx * q * ops.q.D²y') +  (ops.q.D²x * q * ops.q.Wy') * ops.ψ.D²y'

    return term1 + term2 + term3
end

function velocity(q::Matrix{T}, ops::CavityOperators{T, M}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    u = ops.q.Wx * q * ops.q.Dy' 
    v = -ops.q.Dx * q * ops.q.Wy'
    return u, v
end

function vorticity(q::Matrix{T}, ops::CavityOperators{T, M}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    return ops.q.Wx * q * ops.q.D²y' + ops.q.D²x * q * ops.q.Wy'
end


function laplacian_matrix(ops::CavityOperators{T, M}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    return kron(ops.q.Wy, ops.q.D²x) + kron(ops.q.D²y, ops.q.Wx)
end
 
function biharmonic_matrix(ops::CavityOperators{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    # Your existing L
    L = laplacian_matrix(ops)
        
    Ix = I(size(ops.ψ.Dx, 1))
    Iy = I(size(ops.ψ.Dy, 1))

    Lψ = kron(Iy, ops.ψ.D²x) + kron(ops.ψ.D²y, Ix)
    
    return Lψ * L
end

function get_boundary_indices(Ns::Tuple{Int, Int})
    li = LinearIndices((Ns[1]+1, Ns[2]+1))
    
    bottom = li[:, 1]
    top    = li[:, end]
    left   = li[1, :]
    right  = li[end, :]
    
    walls = unique([bottom; left; right])
    return walls, top
end

function build_system_matrix(ops::CavityOperators{T, M}, Δt::T, Re::T) where {T<:AbstractFloat, M <: AbstractMatrix{T}}

    L = laplacian_matrix(ops)
    Bi = biharmonic_matrix(ops)

    A = L - 0.5 * Δt / Re * Bi

    return A

end

function system_matrix_free(q::Vector{T}, ops::CavityOperators{T, M}, Δt::T, Re::T) where {T<:AbstractFloat, M <: AbstractMatrix{T}}

    q_mat = reshape(q, size(ops.q.Dx, 1), size(ops.q.Dx, 1))

    Lq = laplacian(q_mat, ops)
    Bi_q = biharmonic(q_mat, ops)

    res = Lq - 0.5 * Δt / Re * Bi_q

    ## Enforce BCs
    res[1, :] .= q_mat[1, :]
    res[end, :] .= q_mat[end, :]
    res[:, 1] .= q_mat[:, 1]
    res[:, end] .= q_mat[:, end]

    
    return res[:]
end

function apply_bcs_system!(A::M, walls_idx::Vector{Int}, top_idx::Vector{Int}) where {T<:AbstractFloat, M <: AbstractMatrix{T}}
    A[walls_idx, :] .= 0.0
    A[walls_idx, walls_idx] .= I(length(walls_idx))
    A[top_idx, :] .= 0.0
    A[top_idx, top_idx] .= I(length(top_idx)) 
end

function apply_bcs_rhs!(rhs::AbstractVector{T}, walls_idx::Vector{Int}, top_idx::Vector{Int}, grid::ChebyshevGrid{T, 2}) where T<:AbstractFloat
    lid_xs = [x[1] for x in grid.xs[1]]
    rhs[walls_idx] .= 0.0
    rhs[top_idx] .= -1/2 * (1 .- lid_xs.^2) 
end


Δt = 0.0005
Re = 30000 / 2.0

grid = ChebyshevGrid((128, 128), (0.96, 0.96), Float64)
ops = CavityOperators(Matrix{Float64}, grid)

walls_idxs_flat, top_idxs_flat = get_boundary_indices(grid.Ns)

A = build_system_matrix(ops, Δt, Re)

apply_bcs_system!(A, walls_idxs_flat, top_idxs_flat)

A⁻¹ = inv(A)


function step!(q, q_prev, F, ops, grid, Δt, Re)


    rhs = laplacian(q, ops) + 0.5 * Δt/Re * biharmonic(q, ops) - Δt * ( 3/2 * convection(q, ops) - 1/2 * convection(q_prev, ops) )
    
    rhs_vec = view(rhs, :)
    
    apply_bcs_rhs!(rhs_vec, walls_idxs_flat, top_idxs_flat, grid)

    q_prev .= q

    q[:] .= F * rhs_vec

    apply_bcs_rhs!(q[:], walls_idxs_flat, top_idxs_flat, grid) # Enforece BCs strictly
    
end

q = zeros((grid.Ns[1]+1, grid.Ns[2]+1))
q_prev = copy(q)
apply_bcs_rhs!(view(q, :), walls_idxs_flat, top_idxs_flat, grid)

num_steps = 60000

##

for n in 1:num_steps
    step!(q, q_prev, A⁻¹, ops, grid, Δt, Re)
    println("Completed step $n / $num_steps")
    if isnan.(q) |> any
        error("Solution diverged at step $n")
    end
end

##
u, v = velocity(q, ops)

# Resample on a regular grid for plotting
#xs = grid.xs[1]
#ys = grid.xs[2]
xs = range(-1, 1, length=200)
ys = range(-1, 1, length=200)

Ix, Iy = interp_matrices((collect(xs), collect(ys)), grid)

mag =  Ix * sqrt.(u.^2 + v.^2) *  Iy'
#mag = sqrt.(u.^2 + v.^2)

fig = Figure(resolution = (800, 800))
ax = Axis(fig[1, 1], title = "Lid-Driven Cavity Flow (Re = $Re)", xlabel = "x", ylabel = "y")
heatmap!(ax, xs, ys, mag; colormap = :plasma, interpolate = true)
fig

##
ω = vorticity(q, ops)

xs = range(-1, 1, length=200)
ys = range(-1, 1, length=200)

Ix, Iy = interp_matrices((collect(xs), collect(ys)), grid)

ω_plot = Ix * ω * Iy'

fig2 = Figure(resolution = (800, 800))
ax2 = Axis(fig2[1, 1], title = "Vorticity Field (Re = $Re)", xlabel = "x", ylabel = "y")
heatmap!(ax2, xs, ys, ω_plot; colormap = :phase, interpolate = true)
fig2

##
ψ = ops.q.Wx * q * ops.q.Wy'


xs = range(-1, 1, length=200)
ys = range(-1, 1, length=200)

Ix, Iy = interp_matrices((collect(xs), collect(ys)), grid)

ψ_plot = Ix * ψ * Iy'
fig3 = Figure(resolution = (800, 800))
ax3 = Axis(fig3[1, 1], title = "Streamfunction (Re = $Re)", xlabel = "x", ylabel = "y")
contour!(ax3, xs, ys, ψ_plot; colormap = :balance, levels = 50)
fig3