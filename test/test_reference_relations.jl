# Phase 1: what the reference operators are, exactly.
@testset "reference discrete relations" begin
    Random.seed!(11)
    for α in (0.0, 0.96), N in (8, 16)
        grid, ops = setup(N; α)
        q = randn(N+1, N+1)
        Wx, Wy = ops.q.Wx, ops.q.Wy
        D2x, D2y = ops.ψ.D²x, ops.ψ.D²y
        ω = laplacian(q, ops)
        Ψ = Wx * q * Wy
        ΔNΨ = D2x * Ψ + Ψ * D2y'                       # plain P_N Laplacian of nodal ψ
        Δω = D2x * ω + ω * D2y'                        # plain P_N Laplacian of ω

        # laplacian == vorticity == kron matrix
        @test laplacian(q, ops) == vorticity(q, ops)
        @test reshape(laplacian_matrix(ops) * vec(q), N+1, N+1) ≈ ω atol = 1e-12 * norm(ω)
        # biharmonic_matrix is Lψ*L, i.e. EXACTLY Δ_N(laplacian(q))
        @test reshape(biharmonic_matrix(ops) * vec(q), N+1, N+1) ≈ Δω atol = 1e-11 * norm(Δω)
        # A (before BC rows) == (I - cΔ_N) L
        c = 0.5 * Δt_test / Re_test
        Lψ = kron(I(N+1), D2x) + kron(D2y, I(N+1))
        Lm = laplacian_matrix(ops)
        @test build_system_matrix(ops, Δt_test, Re_test) ≈ Lm - c * Lψ * Lm atol = 1e-13 * norm(Lm)
        # ω is the exact Laplacian of the degree-(N+2) polynomial (1-x²)(1-y²)q, which
        # is NOT the P_N Laplacian of its nodal values: the two differ at O(1) for a
        # generic q (aliasing of degree N+1, N+2 modes).
        @test norm(ω - ΔNΨ) > 1e-2 * norm(ω)
        # corners of ω are exactly zero (Wx[1]=Wy[1]=0)
        @test all(ω[[1, end], [1, end]] .== 0)
        # biharmonic(q): equals Δ_N ω exactly only without mapping (α = 0)
        B = biharmonic(q, ops)
        if α == 0
            @test B ≈ Δω atol = 1e-11 * norm(B)
        else
            @test norm(B - Δω) > 1e-3 * norm(B)
        end
    end
    # for a smooth field the P_N and (1-x²)(1-y²)P_N pictures agree spectrally
    # (α = 0: with the arcsin mapping q(x(η)) has singularities at η = ±1/α ≈ ±1.04,
    #  so convergence is only ~1.33⁻ᴺ and N=48 gives ~1e-4; see the note)
    grid, ops = setup(40; α = 0.0)
    q = smooth_q(grid)
    Ψ = ops.q.Wx * q * ops.q.Wy
    @test laplacian(q, ops) ≈ laplacian_ψ(Ψ, ops) rtol = 1e-9
    grid, ops = setup(128)
    q = smooth_q(grid)
    Ψ = ops.q.Wx * q * ops.q.Wy
    @test laplacian(q, ops) ≈ laplacian_ψ(Ψ, ops) rtol = 1e-6
end
