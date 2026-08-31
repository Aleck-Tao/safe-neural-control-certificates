# Analytic Derivation

## 1. Model and sets

The v0.1 plant is

```text
x+ = a*x + b*u + w,
```

with

```text
a in [a_min, a_max],  b > 0,  |w| <= d,
|x| <= X,             |u| <= U.
```

The interval coefficient is constant during a configured rollout, although the one-step certificate remains valid if it changes arbitrarily inside the interval. The disturbance may change adversarially at every step.

The desired practical contraction is

```text
|x+| <= rho*|x| + eta,
0 < rho < 1,
eta = (1-rho)*R,
```

where `R < X` is the target ultimate radius.

## 2. Removing the bounded disturbance

By the triangle inequality,

```text
|a*x + b*u + w| <= |a*x + b*u| + d.
```

It is therefore sufficient to require

```text
|a*x + b*u| <= r(x),
r(x) = rho*|x| + eta - d.
```

The first feasibility condition is

```text
eta - d >= 0,
```

so `r(x)` is non-negative at the origin.

## 3. Robust action interval

For a fixed state, define the uncertain drift endpoints

```text
z_min(x) = min(a_min*x, a_max*x),
z_max(x) = max(a_min*x, a_max*x).
```

The inequality must hold for every drift value in this interval. Since `b > 0`, the robust contraction interval is

```text
u >= (-r(x) - z_min(x))/b,
u <= ( r(x) - z_max(x))/b.
```

It is intersected with `[-U,U]`. The implementation also intersects a directly derived robust one-step state-safety interval. Under the conditions below that safety interval is redundant, because contraction already implies box invariance; retaining it makes the runtime object explicit and independently testable.

The executed action is the Euclidean projection

```text
u = min(max(pi_NN(x), lower(x)), upper(x)).
```

Thus any finite neural output is accepted by the architecture. The guarantee attaches to this composition, not to `pi_NN` alone.

## 4. Non-empty interval conditions

For a symmetric state and input set, it is sufficient to check non-negative `x`. Let

```text
c0 = eta - d,
Delta_a = a_max - a_min.
```

Before actuator intersection, the interval width in next-state units is

```text
2*r(x) - Delta_a*|x|
= (2*rho - Delta_a)*|x| + 2*c0.
```

Because this expression is affine in `|x|`, its minimum on `[0,X]` occurs at an endpoint. The committed verifier checks

```text
min(2*c0, (2*rho-Delta_a)*X + 2*c0) >= 0.
```

The most demanding actuator corner is `x = X`, `a = a_max`, with the action directed inward. Feasibility requires

```text
(a_max-rho)*X <= b*U + c0.
```

The stored input-authority margin is the right side minus the left side.

## 5. Forward invariance and practical convergence

For any projected action and admissible uncertainty/disturbance,

```text
|x+| <= rho*|x| + eta.
```

If `|x| <= X` and `R <= X`, then

```text
rho*|x| + eta
<= rho*X + (1-rho)*R
<= X.
```

Hence `[-X,X]` is forward invariant. Repeated substitution gives

```text
|x[k]| <= rho^k*|x[0]| + eta*(1-rho^k)/(1-rho).
```

Taking the upper limit yields

```text
limsup |x[k]| <= eta/(1-rho) = R.
```

This is practical contraction to a tube, not asymptotic convergence to zero under persistent disturbance.

## 6. Numerical values

For the committed configuration:

```text
a_min = 1.03, a_max = 1.05, b = 0.1,
d = 0.01, X = 1.5, U = 2.0,
rho = 0.92, R = 0.15, eta = 0.012.
```

The independently recomputed margins are:

```text
eta-d                         = 0.002
X-[rho*X+eta]                 = 0.108
b*U+(eta-d)-(a_max-rho)*X     = 0.007
minimum uncertainty width     = 0.004  (next-state units)
```

The finite-grid experiment is an implementation regression check only. These analytic inequalities supply the stated guarantee.
