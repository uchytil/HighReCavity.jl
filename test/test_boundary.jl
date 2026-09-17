@testset "boundary layout and normal derivatives" begin
    Random.seed!(14)
    Nx, Ny = 9, 7
    L = BoundaryLayout(Nx, Ny)
    @test L.m == 2(Ny-1) + 2(Nx-1)
    @test length(L.points) == L.m && allunique(L.points)
    @test all(p -> !(p in L.corners), L.points)
    @test all(p -> p[1] == 1 && 2 <= p[2] <= Ny, L.points[L.left])
    @test all(p -> p[1] == Nx+1, L.points[L.right])
    @test all(p -> p[2] == 1 && 2 <= p[1] <= Nx, L.points[L.bottom])
    @test all(p -> p[2] == Ny+1, L.points[L.top])
    @test all(L.normal_sign[L.left] .== -1) && all(L.normal_sign[L.top] .== 1)
    # gather/scatter round trip; corners handled explicitly
    M = randn(Nx+1, Ny+1)
    v = gather(M, L)
    M2 = zeros(Nx+1, Ny+1); scatter!(M2, v, L; corners = 7.0)
    @test gather(M2, L) == v
    @test all(M2[c] == 7.0 for c in L.corners)
    @test all(M2[2:Nx, 2:Ny] .== 0)
    @test v[L.left] == M[1, 2:Ny] && v[L.top] == M[2:Nx, Ny+1]

    # normal derivative = outward derivative with plain Chebyshev matrices
    grid = ChebyshevGrid((Nx, Ny), (0.96, 0.96), Float64)
    ops = CavityOperators(Matrix{Float64}, grid)
    ψ = randn(Nx+1, Ny+1)
    d = normal_derivative_boundary(ψ, ops, L)
    ψx = ops.ψ.Dx * ψ; ψy = ψ * ops.ψ.Dy'
    @test d[L.left] ≈ -ψx[1, 2:Ny]
    @test d[L.right] ≈ ψx[Nx+1, 2:Ny]
    @test d[L.bottom] ≈ -ψy[2:Nx, 1]
    @test d[L.top] ≈ ψy[2:Nx, Ny+1]
    # with ψ|Γ = 0 the corner normal derivatives vanish identically → corner constraints redundant
    ψ0 = copy(ψ); ψ0[1, :] .= 0; ψ0[end, :] .= 0; ψ0[:, 1] .= 0; ψ0[:, end] .= 0
    ψx = ops.ψ.Dx * ψ0; ψy = ψ0 * ops.ψ.Dy'
    @test maximum(abs.([ψx[c] for c in L.corners])) < 1e-12 && maximum(abs.([ψy[c] for c in L.corners])) < 1e-12

    # lid data: the reference q boundary and the implied wall velocity
    g = lid_q_boundary(grid, L)
    qref = zeros(Nx+1, Ny+1)
    walls, top = get_boundary_indices(grid.Ns)
    apply_bcs_rhs!(view(qref, :), walls, top, grid)
    @test gather(qref, L) == g
    h = lid_normal_derivative_target(grid, ops, L)
    x = grid.xs[1]
    @test h[L.top] ≈ (1 .- x[2:Nx].^2).^2
    @test all(h[[L.left; L.right; L.bottom]] .== 0)
    # ... and it is exactly the reference `velocity(q)` on the lid (u = ψ_y) for ANY interior q
    q = randn(Nx+1, Ny+1); scatter!(q, g, L)
    u, v = velocity(q, ops)
    @test u[2:Nx, Ny+1] ≈ h[L.top] atol = 1e-13
    @test maximum(abs.(u[2:Nx, 1])) < 1e-13 && maximum(abs.(v[1, 2:Ny])) < 1e-13 && maximum(abs.(v[Nx+1, 2:Ny])) < 1e-13
    @test normal_derivative_boundary_q(q, ops, L) ≈ h atol = 1e-13
end
