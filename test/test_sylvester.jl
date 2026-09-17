@testset "separable solvers vs dense" begin
    Random.seed!(13)
    for N in (8, 12, 20), α in (0.0, 0.96)
        grid, ops = setup(N; α)
        li = LinearIndices((N+1, N+1)); int = vec(li[2:N, 2:N]); bnd = setdiff(1:(N+1)^2, int)
        D2x, D2y = ops.ψ.D²x, ops.ψ.D²y
        Lψ = kron(I(N+1), D2x) + kron(D2y, I(N+1))
        f = randn(N+1, N+1); bdata = randn(N+1, N+1)
        for c in (1e-8, 0.01), mode in (:eigen, :schur)
            H = with_logger(NullLogger()) do
                SeparableHelmholtzSolver(ops, c; mode)
            end
            Hd = I - c * Lψ
            ω = copy(bdata); solve_helmholtz_dirichlet!(ω, f, H)
            ωi = Hd[int, int] \ (vec(f)[int] - Hd[int, bnd] * vec(bdata)[bnd])
            @test vec(ω)[int] ≈ ωi rtol = 1e-11
            @test vec(ω)[bnd] == vec(bdata)[bnd]            # boundary untouched
        end
        for mode in (:eigen, :schur)
            P = with_logger(NullLogger()) do
                SeparablePoissonSolver(ops; mode)
            end
            ψ = copy(bdata); solve_poisson_dirichlet!(ψ, f, P)
            ψi = Lψ[int, int] \ (vec(f)[int] - Lψ[int, bnd] * vec(bdata)[bnd])
            @test vec(ψ)[int] ≈ ψi rtol = 1e-10
        end
        # q-form Poisson: recover the interior of a random q from laplacian(q)
        # (:eigen falls back to :schur because Wx⁻¹D²x_q has a complex spectrum; :ceigen is the complex variant)
        for mode in (:schur, :eigen, :ceigen)
            P = with_logger(NullLogger()) do
                SeparableQPoissonSolver(ops; mode)
            end
            q = randn(N+1, N+1); ω = laplacian(q, ops)
            q2 = copy(q); q2[2:N, 2:N] .= 0
            solve_qpoisson_dirichlet!(q2, ω, P)
            @test q2 ≈ q rtol = 1e-10
            @test laplacian(q2, ops)[2:N, 2:N] ≈ ω[2:N, 2:N] rtol = 1e-10
            # dense cross-check via laplacian_matrix
            Lm = laplacian_matrix(ops)
            qi = Lm[int, int] \ (vec(ω)[int] - Lm[int, bnd] * vec(q)[bnd])
            @test vec(q2)[int] ≈ qi rtol = 1e-10
        end
    end
    # generic Sylvester solver: A X + X Bᵀ = G, both modes, non-symmetric A, B
    A = randn(7, 7) - 10I; B = randn(5, 5) - 10I; G = randn(7, 5)
    for mode in (:eigen, :schur, :ceigen)
        S = with_logger(NullLogger()) do
            SylvesterSolver(A, B; mode)
        end
        X = zeros(7, 5); solve!(X, S, G)
        @test A * X + X * B' ≈ G rtol = 1e-11
    end
end
