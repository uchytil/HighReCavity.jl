# Phases 6–8: influence matrix, single linear solve vs the dense inverse.
@testset "influence solver — QForm reproduces A⁻¹·rhs" begin
    Random.seed!(15)
    for N in (8, 16, 32)
        grid, ops = setup(N)
        sys = ReferenceSystem(grid, ops, Δt_test, Re_test)
        S = with_logger(NullLogger()) do
            InfluenceSolver(QForm, grid, ops, Δt_test, Re_test; build_response = true, verbose = false)
        end
        Sc = with_logger(NullLogger()) do
            InfluenceSolver(QForm, grid, ops, Δt_test, Re_test; verbose = false, poisson_mode = :ceigen)
        end
        L = S.layout
        @test S.rank == L.m                         # nonsingular (C ≈ −I + O(c))
        @test S.Cfact isa LU
        @test size(S.C) == (L.m, L.m)
        for trial in 1:3
            rhs = randn(N+1, N+1)
            rhsb = copy(rhs); apply_bcs_rhs!(view(rhsb, :), sys.walls_idxs_flat, sys.top_idxs_flat, grid)
            qref = reshape(sys.Ainv * vec(rhsb), N+1, N+1)
            for method in (:resolve, :response)
                q = zeros(N+1, N+1); ω = zeros(N+1, N+1)
                solve_timestep_linear!(ω, q, rhs, S; method)
                @test q ≈ qref atol = 1e-12 * norm(qref)
                # the interior equations of the reference system hold
                @test norm((sys.A * vec(q) - vec(rhsb))) < 1e-11 * norm(rhsb)
                # ω is exactly laplacian(q) everywhere (interior AND wall), corners zero
                @test ω ≈ laplacian(q, ops) atol = 1e-11 * norm(ω)
                @test all(ω[[1, end], [1, end]] .== 0)
                # boundary of q is the lid data
                @test gather(q, L) == S.g
                # Helmholtz equation for ω at interior nodes
                Δω = laplacian_ω(ω, ops)
                @test (ω - S.c * Δω)[2:N, 2:N] ≈ rhs[2:N, 2:N] atol = 1e-11 * norm(rhs)
                # normal derivative in the polynomial sense equals the lid target exactly
                @test normal_derivative_boundary_q(q, ops, L) ≈ lid_normal_derivative_target(grid, ops, L) atol = 1e-13
            end
            # fast complex-eigen Poisson stage: same solution to ~κ(V)·eps
            q = zeros(N+1, N+1); ω = zeros(N+1, N+1)
            solve_timestep_linear!(ω, q, rhs, Sc)
            @test q ≈ qref atol = 1e-11 * norm(qref)
        end
    end
end

@testset "influence solver — PsiOmegaForm rank structure" begin
    N = 16
    grid, ops = setup(N)
    S = with_logger(NullLogger()) do
        InfluenceSolver(PsiOmegaForm, grid, ops, Δt_test, Re_test; verbose = false)
    end
    L = S.layout
    m = L.m
    @test S.rank == m - 4
    # analytic null space: corner modes, exactly invisible to the interior Helmholtz equation
    D2x, D2y = ops.ψ.D²x, ops.ψ.D²y; ii = 2:N; jj = 2:N
    for (rx, bx, ry, by) in ((L.left, 1, L.bottom, 1), (L.left, 1, L.top, N+1), (L.right, N+1, L.bottom, 1), (L.right, N+1, L.top, N+1))
        ξ = zeros(m); ξ[rx] .= D2y[jj, by]; ξ[ry] .= -D2x[ii, bx]
        @test norm(S.C * ξ) < 1e-13 * S.svals[1] * norm(ξ)
        # null vector of the SVD-based factorization space
        @test norm(S.Cfact.Vnull' * ξ) ≈ norm(ξ) rtol = 1e-10
    end
    @test S.svals[m-4] / S.svals[1] > 1e-3 && S.svals[m-3] / S.svals[1] < 1e-14   # clear gap
    # LS solve: the residual of ∂ₙψ = h lies entirely in the 4-dim left null space
    rhs = randn(N+1, N+1)
    ω = zeros(N+1, N+1); ψ = zeros(N+1, N+1)
    solve_timestep_linear!(ω, ψ, rhs, S)
    r = normal_derivative_boundary(ψ, ops, L) - S.h
    @test norm(S.Cfact.U' * r) < 1e-11 * max(1.0, norm(r))
    @test all(gather(ψ, L) .== 0) && all(ψ[[1, end], [1, end]] .== 0)      # ψ|Γ = 0
    @test laplacian_ψ(ψ, ops)[2:N, 2:N] ≈ ω[2:N, 2:N] rtol = 1e-10        # Poisson
    @test (ω - S.c * laplacian_ω(ω, ops))[2:N, 2:N] ≈ rhs[2:N, 2:N] rtol = 1e-10   # Helmholtz
    # drop-4 reduction gives the same interior ψ, ω (null modes are invisible) up to the
    # different treatment of the 4 unsatisfiable constraints; it is well conditioned
    S4 = with_logger(NullLogger()) do
        InfluenceSolver(PsiOmegaForm, grid, ops, Δt_test, Re_test; verbose = false, reduction = :drop4)
    end
    @test S4.Cfact.kind == :drop4
    ω4 = zeros(N+1, N+1); ψ4 = zeros(N+1, N+1)
    solve_timestep_linear!(ω4, ψ4, rhs, S4)
    r4 = normal_derivative_boundary(ψ4, ops, L) - S4.h
    @test maximum(abs.(r4[setdiff(1:m, [L.left[1], L.left[end], L.right[1], L.right[end]])])) < 1e-9 * max(1.0, norm(S4.h))
end

@testset "influence solver — QForm across c = Δt/(2Re)" begin
    Random.seed!(16)
    N = 16
    grid, ops = setup(N)
    for Re in (15000.0, 100.0, 1.0, 0.01), Δt in (0.0005, 0.05)
        sys = ReferenceSystem(grid, ops, Δt, Re)
        S = with_logger(NullLogger()) do
            InfluenceSolver(QForm, grid, ops, Δt, Re; verbose = false)
        end
        @test S.rank == S.layout.m
        rhs = randn(N+1, N+1)
        rhsb = copy(rhs); apply_bcs_rhs!(view(rhsb, :), sys.walls_idxs_flat, sys.top_idxs_flat, grid)
        qref = reshape(sys.A \ vec(rhsb), N+1, N+1)
        q = zeros(N+1, N+1); ω = zeros(N+1, N+1)
        solve_timestep_linear!(ω, q, rhs, S)
        @test q ≈ qref rtol = 1e-10
    end
end
