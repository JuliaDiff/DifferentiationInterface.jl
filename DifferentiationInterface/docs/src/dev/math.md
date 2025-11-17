# Mathematical model

This page recaps the mathematical model of automatic differentiation used by DI, which justifies how preparation results are constructed.
It is inspired by

- the [documentation](https://chalk-lab.github.io/Mooncake.jl/stable/understanding_mooncake/rule_system/) of [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl)
- [this Discourse answer](https://discourse.julialang.org/t/do-i-understand-enzyme-properly/97760) about [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)

## Setting and hypotheses

Consider a mathematical function $f(x, c, s) = y$ where

-  $x \in \mathcal{X}$ is the active argument (the one being differentiated)
-  $c \in \mathcal{C}$ is a constant argument (corresponds to [`Constant`](@ref) contexts)
-  $s \in \mathcal{S}$ is a scratch argument (corresponds to [`Cache`](@ref) contexts)
-  $y \in \mathcal{Y}$ is the output

In Julia code, some of the input arguments might be mutated, while the output may be written to as well.
Therefore, the proper model is a function $\phi(x_0, c_0, s_0, y_0) = (x_1, c_1, s_1, y_1)$ where $a_0$ is the state of argument $a$ before $f$ is run, while $a_1$ is its state after $a$ is run.

DI makes the following hypotheses on the implementation of $f$ (aka the behavior of $\phi$):

1. The active argument $x$ is not mutated, so $x_1 = x_0$.
2. The constant argument $c$ is not mutated, so $c_1 = c_0$.
3. The initial value of the scratch argument $s_0$ does not matter. It does not affect any of the states $x_1$, $c_1$, $s_1$, $y_1$.
4. The initial value of the output $y_0$ does not matter. It does not affect any of the states $x_1$, $c_1$, $s_1$, $y_1$.

## Forward mode

We want to compute a Jacobian-Vector Product (JVP) $\dot{y} = \left(\frac{\partial f}{\partial x}\right) \dot{x}$ where $\dot{x} \in \mathcal{X}$ is an input tangent.

To do that, we run our AD backend on $\phi$ with input tangents $(\dot{x}_0, \dot{c}_0, \dot{s}_0, \dot{y}_0)$ and obtain $(\dot{x}_1, \dot{c}_1, \dot{s}_1, \dot{y}_1)$.
The value of interest is
$$\dot{y}_1 = \frac{\partial y_1}{\partial x_0} \dot{x}_0 + \frac{\partial y_1}{\partial c_0} \dot{c}_0 + \frac{\partial y_1}{\partial s_0} \dot{s}_0 + \frac{\partial y_1}{\partial y_0} \dot{y}_0$$

Thanks to our hypotheses 3 and 4 on the function's implementation, $\frac{\partial y_1}{\partial s_0} = 0$ and $\frac{\partial y_1}{\partial y_0} = 0$, so we are left with:
$$\dot{y}_1 = \frac{\partial y_1}{\partial x_0} \dot{x_0} + \frac{\partial y_1}{\partial c_0} \dot{c_0}$$

Thus, as long as we set $\dot{c}_0 = 0$, the output tangent $\dot{y}_1$ contains the correct JVP.
Let us now look at $\dot{c}_1$ with the help of hypothesis 2:
$$\dot{c}_1 = \frac{\partial c_1}{\partial x_0} \dot{x}_0 + \frac{\partial c_1}{\partial c_0} \dot{c}_0 + \frac{\partial c_1}{\partial s_0} \dot{s}_0 + \frac{\partial c_1}{\partial y_0} \dot{y}_0 = \dot{c}_0$$

The tangent of $c$ will always be preserved by differentiation.

## Reverse mode

We want to compute a Vector-Jacobian Product (VJP) $\bar{x} = \left(\frac{\partial f}{\partial x}\right)^* \bar{y}$ where $\bar{y} \in \mathcal{Y}$ is an output sensivity.

To do that, we run our AD backend on $\phi$ with output sensitivities $(\bar{x}_1, \bar{c}_1, \bar{s}_1, \bar{y}_1)$ and obtain $(\bar{x}_0, \bar{c}_0, \bar{s}_0, \bar{y}_0)$.
The value of interest is
$$\bar{x}_0 = \left(\frac{\partial x_1}{\partial x_0}\right)^* \bar{x}_1 + \left(\frac{\partial c_1}{\partial x_0}\right)^* \bar{c}_1 + \left(\frac{\partial s_1}{\partial x_0}\right)^* \bar{s}_1 + \left(\frac{\partial y_1}{\partial x_0}\right)^* \bar{y}_1$$

Thanks to our hypotheses 1 and 2 on the function's implementation, $\frac{\partial x_1}{\partial x_0} = I$ and $\frac{\partial c_1}{\partial x_0} = \frac{\partial c_0}{\partial x_0} =0$, so we are left with:
$$\bar{x}_0 = \bar{x}_1 + \left(\frac{\partial s_1}{\partial x_0}\right)^* \bar{s}_1 + \left(\frac{\partial y_1}{\partial x_0}\right)^* \bar{y}_1$$

Thus, as long as we set $\bar{x}_1 = 0$ and $\bar{s}_1 = 0$, the input sensitivity $\bar{x}_0$ contains the correct VJP.
Let us now look at $\bar{s}_0$ with the help of hypothesis 3, which tells us that $\frac{\partial x_1}{\partial s_0} = 0$, $\frac{\partial c_1}{\partial s_0} = 0$, $\frac{\partial s_1}{\partial s_0} = 0$, and $\frac{\partial y_1}{\partial s_0} = 0$:

$$\bar{s}_0 = \left(\frac{\partial x_1}{\partial s_0}\right)^* \bar{x}_1 + \left(\frac{\partial c_1}{\partial s_0}\right)^* \bar{c}_1 + \left(\frac{\partial s_1}{\partial s_0}\right)^* \bar{s}_1 + \left(\frac{\partial y_1}{\partial s_0}\right)^* \bar{y}_1 = 0$$

The sensitivity of $s$ will always be set to $0$ by differentiation.

## Implementation

DI's preparation mechanism allows pre-allocating the memory for tangents and sensitivities, inside a `prep` object.
This object is then reused across several AD calls.

For mutable objects, each AD call performs the following transformations on the provided shadow/dual storage (`Duplicated` for Enzyme, `Dual` / `CoDual` for Mooncake):

- In forward mode, $\dot{a}$ is updated from $\dot{a}_0$ to $\dot{a}_1$
- In reverse mode, $\bar{a}$ is updated from $\bar{a}_1$ to $\bar{a}_0$

### At initialization

How to initialize shadow/dual memory inside `prep`?

- In forward mode, make sure that $\dot{c} = 0$.
- In reverse mode, make sure that $\bar{x} = 0$ and $\bar{s} = 0$.

### At every call

Should the shadow/dual memory inside `prep` be reset before every AD call?

- In forward mode, no need ($\dot{c}$ will remain $0$ if it is initialized to $0$)
- In reverse mode, just set $\bar{x} = 0$ ($\bar{s}$ will be reset to $0$ at every AD call)
