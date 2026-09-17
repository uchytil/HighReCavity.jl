module HighReCavity

using LinearAlgebra
using Printf

include("reference.jl")
include("psiomega.jl")
include("sylvester.jl")
include("boundary.jl")
include("influence.jl")
include("timestep.jl")

# reference (verbatim) API
export ChebyshevGrid, CavityOperators, diff_matrices, laplacian, biharmonic, convection,
       velocity, vorticity, laplacian_matrix, biharmonic_matrix, get_boundary_indices,
       build_system_matrix, apply_bcs_system!, apply_bcs_rhs!, interp_matrices
export ReferenceSystem, step_reference!, reference_rhs, reference_initial_state
# ψ–ω routines
export streamfunction_from_q, velocity_from_streamfunction, velocity_from_streamfunction!,
       vorticity_derivatives, vorticity_derivatives!, convection_from_uvω, convection_from_ψω,
       laplacian_ψ, laplacian_ω, laplacian_ψ!, laplacian_ω!, laplacian_boundary!
# separable solvers
export SylvesterSolver, solve!, SeparableHelmholtzSolver, solve_helmholtz_dirichlet!,
       SeparablePoissonSolver, solve_poisson_dirichlet!, SeparableQPoissonSolver, solve_qpoisson_dirichlet!
# boundary
export BoundaryLayout, gather, gather!, scatter!, normal_derivative_boundary, normal_derivative_boundary!,
       normal_derivative_boundary_q, lid_q_boundary, lid_normal_derivative_target
# influence
export Formulation, QForm, PsiOmegaForm, InfluenceSolver, build_influence_matrix!, factorize_influence!,
       solve_timestep_linear!
# time stepping
export QWorkspace, laplacian!, biharmonic!, convection!, qform_rhs!, QState, step_influence!,
       PsiOmegaState, psiomega_rhs!, step_psiomega!

end # module
