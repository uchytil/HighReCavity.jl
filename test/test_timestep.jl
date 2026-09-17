# Phase 8: several actual simulation steps, old vs new solver, from the same physical state.
@testset "time stepping — influence vs dense reference" begin
    for N in (32, 64)
        grid, ops = setup(N)
        sys = ReferenceSystem(grid, ops, Δt_test, Re_test)
        S = with_logger(NullLogger()) do
            InfluenceSolver(QForm, grid, ops, Δt_test, Re_test; build_response = true, verbose = false)
        end
        L = S.layout
        q, qp = reference_initial_state(sys)
        stA = QState(S); stB = QState(S)
        @test stA.q == q && stA.q_prev == qp
        nsteps = 20
        for n in 1:nsteps
            step_reference!(q, qp, sys)
            step_influence!(stA, S; method = :resolve)
            step_influence!(stB, S; method = :response)
        end
        # Agreement is limited by conditioning: κ(A) ≈ κ(Δ_N) ~ 1e4 (N=32) … 1e5 (N=64), and the
        # reference `inv(A)*rhs` itself differs from `A\\rhs` at the κ·eps level (see
        # scripts/compare_reference.jl for the measured floor).
        for st in (stA, stB)
            tol = N == 32 ? 1e-10 : 1e-9
            @test st.q ≈ q atol = tol * norm(q)
            @test st.q_prev ≈ qp atol = tol * norm(qp)
            # derived quantities
            @test streamfunction_from_q(st.q, ops) ≈ streamfunction_from_q(q, ops) atol = tol * norm(streamfunction_from_q(q, ops))
            @test st.ω ≈ laplacian(q, ops) atol = tol * norm(laplacian(q, ops))
            u1, v1 = velocity(st.q, ops); u0, v0 = velocity(q, ops)
            @test u1 ≈ u0 atol = tol * norm(u0)
            @test v1 ≈ v0 atol = tol * norm(v0)
            @test convection(st.q, ops) ≈ convection(q, ops) atol = tol * norm(convection(q, ops))
            @test gather(st.q, L) == S.g
            @test normal_derivative_boundary_q(st.q, ops, L) ≈ lid_normal_derivative_target(grid, ops, L) atol = 1e-13
            # nodal ψ vanishes on the boundary
            @test all(gather(streamfunction_from_q(st.q, ops), L) .== 0)
        end
    end
end
