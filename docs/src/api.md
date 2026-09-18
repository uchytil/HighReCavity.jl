# [API](@id api)

```@docs
HighReCavity
```

## Simulation

```@docs
CavityParameters
CavitySimulation
step!
run!
```

## Fields

```@docs
streamfunction
vorticity
velocity
grid
```

## Discretisation

```@docs
ChebyshevGrid
diff_matrices
interp_matrix
CavityOperators
laplacian!
laplacian_nodal!
biharmonic!
convection!
rhs!
reynolds_internal
```

## Time integrators

```@docs
HighReCavity.ark3_step!
HighReCavity.implicit_weight
```

## Implicit solve

```@docs
InfluenceSolver
influence_solve!
HighReCavity.BoundaryLayout
HighReCavity.HelmholtzSolver
HighReCavity.QPoissonSolver
HighReCavity.wall_laplacian!
HighReCavity.lid_boundary_values
HighReCavity.set_walls!
HighReCavity.SylvesterSolver
HighReCavity.sylvester_solve!
HighReCavity.diffusion_coefficient
```
