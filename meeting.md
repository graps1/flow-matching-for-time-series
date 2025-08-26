**Organizational stuff**

- ICLR deadline Sep 19th
- Vacation starting Sep 10th
- => a bit more than two weeks left

**Summary**

- I have successfully trained a bunch of flow matching models on:
    - the 2d Kuramoto-Sivashinsky equation, size 256 x 256
    - a compressible 2d Navier-Stokes setting w/ periodic BCs, size 64 x 64
    - the full 3d Rayleigh-Taylor instability compressed to a size of 32 x 32 x 32
    - slices of the 3d Rayleigh-Taylor instability, size 128 x 128
- Contributions:
    - a novel distillation method (i.e., a method that directly learns the solution of the flow matching ODE) for faster output generation
    - a (physics-informed) Sobolev loss that avoids blurry outputs

**Main argument**

Using flow matching (or denoising diffusion) is really promising for the surrogate modeling of (partially observed) systems

- quality-wise, these methods are capable of dealing w/ missing information and naturally generate plausible successor states.
- in the case of fully deterministic systems, the flow matching ODE can -- in theory -- be solved with a single iteration of explicit Euler, since the transition distribution is a Dirac delta centered at the successor and the flow matching paths are therefore completely straight. => In this case, a flow matching model is as efficient as a deterministic surrogate model(!)

But: 

1. for non-deterministic systems, one needs more solver steps. About 10 explicit Euler / 5 midpoint steps usually suffice, but this is still ~5x slower than a single-step model.
1. (this has nothing to do with the previous argument:) results often look blurry due to the use of the l2 norm for training. This is something that we want to avoid, since sharp features frequently appear in fluid dynamics applications.

So:

- Problem 1) motivates the use of a distillation-based approach.
- Problem 2) motivates the use of a different loss function, in particular a Sobolev loss that incorporates (coordinate-space) gradients.

**Preliminaries**

In flow matching, one learns the flow taking samples from the initial Gaussian distribution $p_0$ to the target distribution $p_1$. This is done by solving the optimization problem
$$
    \min_\theta \mathbb E_{p_0(\bm x_0), p_1(\bm x_1), U(\bm t)} \lVert v_t^\theta(\bm t \bm x_1 + (1-\bm t)\bm x_0) - (\bm x_1 - \bm x_0) \rVert^2.
$$
In our case, $\bm x_1$ is a fluid field consisting of velocities and a pressures, perhaps also densities. In order to do prediction, i.e., in order to move from one fluid state to the next, we use a conditional model of the form
$$
    \min_\theta \mathbb E_{p_0(\bm x_0), p_1(\bm x_1^-, \bm x_1^+), U(\bm t)} \lVert v_t^\theta(\bm t \bm x_1^+ + (1-\bm t)\bm x_0 ∣ \bm x_1^-) - (\bm x_1^+ - \bm x_0) \rVert^2,
$$
where $\bm x_1^+$ is now the next and $\bm x_1^-$ the previous fluid state. Bot $\bm x_1^-$ and $\bm x_1^+$ are drawn from the same trajectory.

### 1. The distillation method 

Say we have trained a flow matching velocity model $v_t(x_t) = v_t^\theta(x_t)$, which is assumed wlog to be unconditional. To generate new approx. samples from $p_1$, one samples $x_0 ∼ p_0$ from the Gaussian distribution and then solve the ODE
$$
    \dot {x}_t = v_t(x_t), \quad\quad(*)
$$
from $t = 0$ to $t = 1$. Associated with the velocity field $v^\theta$ is the underlying *flow* that it generates:
$$
    F_\delta(x_t, t) = x_{t + \delta}
$$
when $x_t$ follows the ODE $(*)$. If we *had* access to a model computing $F$, we could compute the solution of $x_0$ in a single step by evaluating $F_1(x_0, 0)$.

But we don't have access to it, so we have to learn it. The first insight is that we can characterize the flow by three properties:

1. (identity when no increment) $F_0(x_t, t) = x_t$.
1. (consistency w/ velocity field) $\frac d {d\delta} F^\xi_\delta(x_t, t) |_{\delta = 0} = v_t(x_t)$.
1. (semigroup property) $F_{a+b}(x_t, t) = F_a(F_b(x_t,t), t+b)$.

Then, we parametrize a neural network:
$$
    F^\xi_\delta(x_t, t) = \underbrace{x_t + \delta v_t(x_t)}_{\text{explicit Euler step}} + \underbrace{\delta^2 (\phi^\xi_\delta(x_t,t) - v_t(x_t))}_{\text{learned correction}}.
$$

This already ensures properties 1) and 2). To get the third property, we are optimizing the following "semigroup" loss:
$$
    \min_\xi \mathbb E_{\bm \delta, \bm t, \bm x_1, \bm x_0}  \lVert F_{\bm \delta}^\xi(\bm x_{\bm t}, \bm t) - \text{sg}(F_{\bm \delta/2}^\xi(F_{\bm \delta/2}^\xi(\bm x_{\bm t}, \bm t), \bm t + \bm \delta/2)) \rVert^2.
$$
Here, the $\text{sg}$ is the "stopgrad" operation. I'm utilizing it here since the rhs is a more accurate version of what we're trying to learn, i.e. closer to the true flow $F_{\bm \delta}(\bm x_{\bm t}, \bm t)$.
In practice, we are initializing $\phi$ (the "corrector" network) with a copy of $v$, where we modify the input weights to add an additional channel for $\delta$.


## 2. Avoiding blurryness in generated states 

- When being trained, both the velocity model (trained w/ flow matching) and the distilled flow (trained w/ semigroup loss) tend to produce blurry results.
- This is likely due to the use of the l2 norm in image space.

The Sobolev norm addresses this issue: In our case, $x_1$ is a fluid field, i.e., it is a differentiable function of the form $x_1 : \R^d \rightarrow \R^n$, where $d$ is the coordinate dimension ($d  = 2$ or $d = 3$) and $n$ is the number of fields, e.g. $n = 3$ when there is one pressure and a 2d velocity field.

One can define a weighted Sobolev norm by taking
$$
    \lVert x_1 \rVert^2_S = \alpha \lVert x_1 \rVert^2_2 + \beta \lVert \nabla x_1 \rVert^2_2,
$$
where $\alpha$ and $\beta$ are constants. Since it includes the gradient of $x_1$, it is more sensitive to changes of $x_1$ in the coordinate domain. In other words, the difference $\lVert x_1 - x_1' \rVert^2_S$ is now larger if the gradients of $x_1$ and $x_1'$ don't match, which is the case if $x_1$ is, for instance, a blurry version of $x_1'$.

This norm can be extended to the case where $x_t$ is a not fully denoised fluid field:
$$
    \lVert x_t \rVert^2_{S_t} = \alpha_t \lVert x_t \rVert^2_2 + \beta_t \lVert \nabla x_t \rVert^2_2,
$$
where $\alpha_t$ and $\beta_t$ are now allowed to depend on the time, e.g., we can let $\alpha_t$ stay constant and increase $\beta_t$ as $t \rightarrow 1$ in order to put a larger priority on "deblurring" less noisy fluid fields. 

The **first** way one can use this norm is to train the velocity model with this new norm instead:
$$
    \min_\theta \mathbb E_{p_0(\bm x_0), p_1(\bm x_1), U(\bm t)} \lVert v_t^\theta(\bm t \bm x_1 + (1-\bm t)\bm x_0) - (\bm x_1 - \bm x_0) \rVert^2_{S_{\bm t}}.
$$
This is actually sound, in the sense that $v^\theta$ is learning the correct velocity field.

The **second** way one can use this is to improve training of the distillation model: The new objective simply becomes:
$$
    \min_\xi \mathbb E_{\bm \delta, \bm t, \bm x_1, \bm x_0}  \lVert F_{\bm \delta}^\xi(\bm x_{\bm t}, \bm t) - \text{sg}(F_{\bm \delta/2}^\xi(F_{\bm \delta/2}^\xi(\bm x_{\bm t}, \bm t), \bm t + \bm \delta/2)) \rVert^2_{S_{\bm t}}.
$$