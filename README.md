# HighReCavity.jl

A Chebyshev collocation solver for the two-dimensional lid-driven cavity at high Reynolds
number, in streamfunction–vorticity form. Long time integrations are the intended use: the
solver keeps only the current state and processes it through a callback.

```julia
using HighReCavity

params = CavityParameters(N = 128, Re = 30_000, dt = 5e-4, alpha = 0.96, backend = :ceigen)
sim = CavitySimulation(params)

run!(sim, 20_000)                                   # transient
run!(sim, 40_000; every = 200, callback = process)  # production interval, process(sim) every 200 steps

ψ = streamfunction(sim);  ω = vorticity(sim);  u, v = velocity(sim)
```

## 1. Problem

Incompressible flow in the square cavity $\Omega = [-1,1]^2$ driven by the top wall. In
streamfunction–vorticity variables, with $u = \psi_y$, $v = -\psi_x$ and $\omega = \Delta\psi$,

$$
\partial_t \omega + u\,\omega_x + v\,\omega_y = \nu\,\Delta\omega, \qquad \Delta\psi = \omega .
$$

The boundary conditions are no penetration, $\psi = 0$ on all four walls, and the tangential
velocity

$$
u(x, 1) = (1-x^2)^2 \quad\text{(lid)}, \qquad u = v = 0 \text{ on the other walls},
$$

i.e. a regularised lid whose velocity vanishes at the corners, so the corner singularity of
the classical uniformly translating lid is absent. The fluid starts at rest and the lid is
started impulsively.

## 2. Reynolds number

`Re` is the Reynolds number $Re = UL/\nu$ based on the full cavity side length $L = 2$ and the
peak lid speed $U = 1$. `Re = 30_000` is a Reynolds-number-30 000 cavity. In the units of the
equations above (domain $[-1,1]^2$, $U = 1$) this means $\nu = 2/Re$.

## 3. Spatial discretisation

**Grid.** Each direction uses $N+1$ Chebyshev–Gauss–Lobatto nodes $\eta_k = -\cos(\pi k/N)$,
$k = 0,\dots,N$, mapped to the physical coordinate by

$$
x = \frac{\arcsin(\alpha\,\eta)}{\arcsin\alpha}, \qquad 0 \le \alpha < 1 .
$$

The Gauss–Lobatto nodes cluster like $O(N^{-2})$ at the walls; the mapping spreads them out
(spacing $\to O(N^{-1})$ as $\alpha \to 1$), which relaxes the time-step restriction of the
explicit convection term. Since the mapping is analytic only for $|\alpha\eta| < 1$, $\alpha$
close to 1 trades some of the spectral convergence rate for this. `alpha = 0` gives the
unmapped Chebyshev grid; the default is `alpha = 0.96`. The same grid is used in $x$ and $y$.

Derivatives are collocation derivatives: the standard Chebyshev differentiation matrix in
$\eta$ scaled by $d\eta/dx$ gives $D_1$, and $D_2 = D_1 D_1$.

**Streamfunction representation.** The state variable is $q$, with

$$
\psi(x,y) = (1-x^2)(1-y^2)\, q(x,y) .
$$

The prefactor makes $\psi = 0$ on all walls for every $q$, and it places the wall-normal
derivative of $\psi$ in the wall values of $q$:

$$
\psi_y(x, \pm 1) = \mp 2\,(1-x^2)\, q(x, \pm 1), \qquad
\psi_x(\pm 1, y) = \mp 2\,(1-y^2)\, q(\pm 1, y).
$$

The velocity boundary conditions therefore become Dirichlet conditions on $q$:
$q = 0$ on the left, right and bottom walls and $q(x,1) = -\tfrac12 (1-x^2)$ on the lid, which
gives exactly $u(x,1) = (1-x^2)^2$.

**Operators.** Fields are $(N+1)\times(N+1)$ arrays $Q_{ij} = q(x_i, y_j)$; $x$-derivatives act
from the left ($D\,Q$) and $y$-derivatives from the right ($Q\,D^{\mathsf T}$). Derivatives of
$\psi$ are formed by the product rule applied to $(1-x^2)\,q(x)$, with $q$ differentiated by
the collocation matrices. With $W = \mathrm{diag}(1-x_i^2)$ and $X = \mathrm{diag}(x_i)$,

$$
\begin{aligned}
\partial_x\big[(1-x^2)q\big] &= (1-x^2)q' - 2xq &&\to\ D_q = W D_1 - 2X,\\
\partial_x^2\big[(1-x^2)q\big] &= (1-x^2)q'' - 4xq' - 2q &&\to\ D_{2q} = W D_2 - 4X D_1 - 2I,\\
\partial_x^4\big[(1-x^2)q\big] &= (1-x^2)q'''' - 8xq''' - 12q'' &&\to\ D_{4q} = W D_2^2 - 8X D_2 D_1 - 12 D_2 .
\end{aligned}
$$

The physical fields are then

$$
\Psi = W Q W, \qquad
U = W Q D_q^{\mathsf T}, \qquad
V = -D_q Q W, \qquad
\Omega = L Q := D_{2q} Q W + W Q D_{2q}^{\mathsf T}\ (= \Delta\psi),
$$

and $\Delta^2\psi = D_{4q} Q W + W Q D_{4q}^{\mathsf T} + D_2 (W Q D_{2q}^{\mathsf T}) + (D_{2q} Q W) D_2^{\mathsf T}$.
These are derivatives of the polynomial $\psi$ itself, including on the walls (where the
$W$-weighted terms vanish and only $-2xq$, $-4xq' - 2q$, … remain).

## 4. Time integration

The vorticity equation is advanced with Crank–Nicolson for diffusion and second-order
Adams–Bashforth for convection. With $c = \nu\,\mathrm{dt}/2$ and $\mathcal N = u\,\omega_x + v\,\omega_y$,

$$
(I - c\Delta)\,\omega^{n+1} = f, \qquad
f = \omega^n + c\,\Delta\omega^n - \mathrm{dt}\left(\tfrac32 \mathcal N^n - \tfrac12 \mathcal N^{n-1}\right),
$$

where $\omega^n = LQ^n$ and $\Delta\omega^n = \Delta^2\psi^n$ are formed from $q^n$ with the
operators above, $\omega_x = D_1\Omega$, $\omega_y = \Omega D_1^{\mathsf T}$, and on the
left-hand side $\Delta$ is the collocation Laplacian $D_2\,\Omega + \Omega\,D_2^{\mathsf T}$
acting on the nodal vorticity. On the first step $\mathcal N^{-1} = 0$. The new state $q^{n+1}$
is defined by $L q^{n+1} = \omega^{n+1}$ together with the wall values of $q$; solving this
coupled problem is the subject of the next section. One step is therefore

$$
(q^n, q^{n-1}) \;\to\; f \;\to\; \text{implicit solve} \;\to\; q^{n+1}.
$$

The step size `dt` is limited by the explicit convection term; the diffusion term is
unconditionally stable.

## 5. The implicit solve

The implicit step $(I - c\Delta)\,L q = f$ is solved with the vorticity $\omega = Lq$ as an
auxiliary unknown. Writing $\Gamma$ for the walls and $\xi = \omega|_\Gamma$ for the (unknown)
wall vorticity, the step splits into

$$
\begin{aligned}
&(I - c\Delta)\,\omega = f && \text{at interior nodes},\quad \omega|_\Gamma = \xi, \\
&L q = \omega && \text{at interior nodes},\quad q|_\Gamma = g \ \text{(lid data)}, \\
&(L q)|_\Gamma = \xi && \text{consistency of the wall vorticity}.
\end{aligned}
$$

For a given $\xi$ the first two are Dirichlet problems with separable operators, and both are
solved as Sylvester equations on the $(N-1)^2$ interior nodes:

* **Helmholtz stage.** With $\Omega_i$ the interior block and the known wall rows/columns moved
  to the right-hand side,
  $(\tfrac12 I - c D_{2,ii})\,\Omega_i + \Omega_i\,(\tfrac12 I - c D_{2,ii})^{\mathsf T} = f_i + c\,(D_{2,ib}\,\Omega_{bi} + \Omega_{ib}\,D_{2,ib}^{\mathsf T})$.
* **q-Poisson stage.** $LQ = \Omega$ at interior nodes, after scaling by $W_{ii}^{-1}$ on both sides,
  $(W^{-1}D_{2q})_{ii}\,Q_i + Q_i\,(W^{-1}D_{2q})_{ii}^{\mathsf T} = W_{ii}^{-1}\big(\Omega_i - D_{2q,ib}\,Q_{bi}\,W_{ii} - W_{ii}\,Q_{ib}\,D_{2q,ib}^{\mathsf T}\big)W_{ii}^{-1}$.

Each Sylvester equation $AX + XA^{\mathsf T} = G$ is solved through a decomposition of the
$(N-1)\times(N-1)$ matrix $A$ computed once (Section 6); a solve costs four matrix products
(plus a quasi-triangular Sylvester solve for `:schur`).

The solution of the two stages is affine in $\xi$: $(Lq)|_\Gamma = d_0 + C\xi$, where $d_0$
comes from the solve with $\xi = 0$ and the $m \times m$ *influence matrix* $C$ collects the wall
vorticity produced by unit wall data. The consistency condition becomes

$$
(C - I)\,\xi = -d_0 .
$$

$C - I$ is assembled once at initialisation from the $m = 4(N-1)$ unit responses
$\xi = e_k$ (each a Helmholtz solve with $f = 0$ followed by a q-Poisson solve with $q|_\Gamma = 0$)
and LU-factorised; it equals $-I + O(c)$. The wall unknowns are the non-corner wall nodes: no
interior collocation equation involves a corner value, $Lq$ vanishes identically at the
corners, and the corner values of $q$ enter no equation.

One implicit solve is then

1. solve both stages with $\xi = 0$ and evaluate $d_0 = (Lq_0)|_\Gamma$;
2. solve $(C - I)\,\xi = -d_0$;
3. solve both stages again with $\omega|_\Gamma = \xi$.

The result is $q^{n+1}$ with its wall values equal to the lid data, and $\omega^{n+1} = Lq^{n+1}$
at every node. The no-slip condition never appears as a separate constraint: it is carried by
$q|_\Gamma = g$, since $\partial_n\psi = -2(1-x^2)\,q|_\Gamma$ in this representation.

## 6. Solver backends

`backend` selects how the q-Poisson Sylvester equation is solved; the Helmholtz operator
$\tfrac12 I - cD_2$ has a real, well-conditioned spectrum and always uses an eigendecomposition.

* `:ceigen` (default) — eigendecomposition $A = V\Lambda V^{-1}$,
  $X = V\big[(V^{-1} G V^{-\mathsf T}) ./ (\lambda_i + \lambda_j)\big]V^{\mathsf T}$. The
  operator $W^{-1}D_{2q}$ has complex eigenvalues in its highest modes, so this stage runs in
  complex arithmetic; the accuracy is governed by $\kappa(V)\,\varepsilon$ with
  $\kappa(V) \sim 10^3$.
* `:schur` — real Schur form and Bartels–Stewart (LAPACK `trsyl`). Backward stable
  independently of $\kappa(V)$; several times slower because `trsyl` is unblocked. Use it as
  a cross-check or when conditioning is a concern.

Both backends solve the same discrete problem; the test suite checks that they agree.

## 7. Usage

```julia
params = CavityParameters(
    N       = 128,      # polynomial degree, (N+1)² nodes
    Re      = 30_000,   # Reynolds number, full side length L = 2
    dt      = 5e-4,     # time step
    alpha   = 0.96,     # grid mapping parameter (0 = unmapped)
    backend = :ceigen,  # :ceigen or :schur
)
sim = CavitySimulation(params)   # grid, operators, decompositions, influence matrix (fixed cost)

step!(sim)          # one time step
run!(sim, 60_000)   # many steps
sim.step, sim.t     # steps taken, current time
```

`CavityParameters` also accepts `T = Float32` etc. to choose the floating-point type. With the
default backend, initialisation takes about 0.4 s at N = 128 and 4 s at N = 256, and one step
about 2 ms and 10 ms respectively (Apple M2 Max, 8 BLAS threads).

## 8. Post-processing

```julia
sim.q                 # state q,  ψ = (1−x²)(1−y²) q
streamfunction(sim)   # Ψ = W q W
vorticity(sim)        # ω = Δψ
velocity(sim)         # (u, v) = (ψ_y, −ψ_x)
grid(sim).x           # node coordinates (same in x and y)
```

All fields are `(N+1)×(N+1)` arrays indexed `[i, j] ↔ (x_i, y_j)`. `sim.ω` holds the vorticity
produced by the last implicit solve, which equals `vorticity(sim)`. The same functions accept
`(q, ops)` directly, e.g. `velocity(q, sim.ops)`, for states saved earlier. To evaluate a field
on a uniform grid for plotting:

```julia
xs = range(-1, 1, length = 201)
M  = interp_matrix(collect(xs), grid(sim))     # barycentric interpolation matrix
Ψ_plot = M * streamfunction(sim) * M'
```

## 9. Long simulations

`run!` keeps only the current state. Periodic output goes through the callback, which is
called with the simulation after every `every`-th step (and after the last one):

```julia
run!(sim, 20_000)                                          # transient, nothing recorded

using Serialization
energy = Float64[]
function collect!(s)
    u, v = velocity(s)
    push!(energy, sum(abs2, u) + sum(abs2, v))
    s.step % 5000 == 0 && serialize("q_$(s.step).jls", (q = s.q, q_prev = s.q_prev, t = s.t))
end
run!(sim, 1_000_000; every = 200, callback = collect!)      # production interval
```

`run!` stops with an error if the solution becomes NaN (`check_nan = false` disables the
check). A run can be resumed from a saved `(q, q_prev)` pair by copying them into `sim.q` and
`sim.q_prev` of a simulation built with the same parameters. See `examples/run_cavity.jl`.

## Layout and tests

```
src/chebyshev.jl    mapped Gauss–Lobatto grid, differentiation, interpolation
src/operators.jl    spectral operators of the q representation, physical fields
src/sylvester.jl    Helmholtz and q-Poisson interior solves as Sylvester equations
src/influence.jl    wall-node layout, influence matrix, one implicit solve
src/simulation.jl   parameters, state, step!, run!
```

`julia --project=. -e 'using Pkg; Pkg.test()'` checks the implicit solve against a direct
dense solution of the same discrete equations at small N, short trajectories against stored
reference data, the boundary conditions, and the agreement of the two backends.

## License

MIT — see [LICENSE](LICENSE).
