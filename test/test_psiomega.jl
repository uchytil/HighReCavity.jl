@testset "ψ–ω routines" begin
    Random.seed!(12)
    grid, ops = setup(24)
    N = 24
    # exact relations (same arithmetic as the reference)
    q = randn(N+1, N+1)
    ω = laplacian(q, ops)
    u, v = velocity(q, ops)
    @test convection_from_uvω(u, v, ω, ops) ≈ convection(q, ops) rtol = 1e-13
    ωx, ωy = vorticity_derivatives(ω, ops)
    @test ωx ≈ ops.ψ.Dx * ω && ωy ≈ ω * ops.ψ.Dy'
    ωx2 = similar(ω); ωy2 = similar(ω); vorticity_derivatives!(ωx2, ωy2, ω, ops)
    @test ωx2 == ωx && ωy2 == ωy
    Ψ = streamfunction_from_q(q, ops)
    @test Ψ == ops.q.Wx * q * ops.q.Wy
    uu, vv = velocity_from_streamfunction(Ψ, ops)
    @test uu ≈ Ψ * ops.ψ.Dy' && vv ≈ -(ops.ψ.Dx * Ψ)
    u2 = similar(Ψ); v2 = similar(Ψ); velocity_from_streamfunction!(u2, v2, Ψ, ops)
    @test u2 ≈ uu && v2 ≈ vv
    out = similar(Ψ); laplacian_ψ!(out, Ψ, ops)
    @test out ≈ laplacian_ψ(Ψ, ops)
    @test laplacian_ω(ω, ops) == laplacian_ψ(ω, ops)
    # boundary-only laplacian
    L = BoundaryLayout(N, N)
    d = zeros(L.m); laplacian_boundary!(d, q, ops, L)
    @test d ≈ gather(ω, L) rtol = 1e-12

    # spectral agreement with the q-based routines on a smooth field (α = 0, see test_reference_relations)
    grid, ops = setup(40; α = 0.0); N = 40
    q = smooth_q(grid)
    Ψ = streamfunction_from_q(q, ops)
    u, v = velocity(q, ops)
    uu, vv = velocity_from_streamfunction(Ψ, ops)
    @test uu ≈ u rtol = 1e-9
    @test vv ≈ v rtol = 1e-9
    ω = laplacian(q, ops)
    @test convection_from_ψω(Ψ, ω, ops) ≈ convection(q, ops) rtol = 1e-8
    @test laplacian_ψ(Ψ, ops) ≈ ω rtol = 1e-9
end
