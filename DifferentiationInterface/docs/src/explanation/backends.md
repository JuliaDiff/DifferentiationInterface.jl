# Backends

## List

We support the following dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

- [`AutoChainRules`](@extref ADTypes.AutoChainRules)
- [`AutoDiffractor`](@extref ADTypes.AutoDiffractor)
- [`AutoEnzyme`](@extref ADTypes.AutoEnzyme)
- [`AutoFastDifferentiation`](@extref ADTypes.AutoFastDifferentiation)
- [`AutoFiniteDiff`](@extref ADTypes.AutoFiniteDiff)
- [`AutoFiniteDifferences`](@extref ADTypes.AutoFiniteDifferences)
- [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff)
- [`AutoGTPSA`](@extref ADTypes.AutoGTPSA)
- [`AutoMooncake`](@extref ADTypes.AutoMooncake)
- [`AutoPolyesterForwardDiff`](@extref ADTypes.AutoPolyesterForwardDiff)
- [`AutoReverseDiff`](@extref ADTypes.AutoReverseDiff)
- [`AutoSymbolics`](@extref ADTypes.AutoSymbolics)
- [`AutoTracker`](@extref ADTypes.AutoTracker)
- [`AutoZygote`](@extref ADTypes.AutoZygote)

## Features

Given a backend object, you can use:

- [`check_available`](@ref) to know whether the required AD package is loaded
- [`check_inplace`](@ref) to know whether the backend supports in-place functions (all backends support out-of-place functions)

In theory, all we need from each backend is either a `pushforward` or a `pullback`: we can deduce every other operator from these two.
In practice, many AD backends have custom implementations for high-level operators like `gradient` or `jacobian`, which we reuse whenever possible.

!!! details
    In the rough summary table below,

    - ✅ means that we reuse the custom implementation from the backend;
    - ❌ means that a custom implementation doesn't exist, so we use our default fallbacks;
    - 🔀 means it's complicated or not done yet.

    |                            | `pf` | `pb` | `der` | `grad` | `jac` | `hess` | `hvp` | `der2` |
    | -------------------------- | ---- | ---- | ----- | ------ | ----- | ------ | ----- | ------ |
    | `AutoChainRules`           | ❌    | ✅    | ❌     | ❌      | ❌     | ❌      | ❌     | ❌      |
    | `AutoDiffractor`           | ✅    | ❌    | ❌     | ❌      | ❌     | ❌      | ❌     | ❌      |
    | `AutoEnzyme` (forward)     | ✅    | ❌    | ❌     | ✅      | ✅     | ❌      | ❌     | ❌      |
    | `AutoEnzyme` (reverse)     | ❌    | ✅    | ❌     | ✅      | ✅     | ❌      | 🔀     | ❌      |
    | `AutoFastDifferentiation`  | ✅    | ✅    | ✅     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoFiniteDiff`           | 🔀    | ❌    | ✅     | ✅      | ✅     | ✅      | ❌     | ❌      |
    | `AutoFiniteDifferences`    | 🔀    | ❌    | ❌     | ✅      | ✅     | ❌      | ❌     | ❌      |
    | `AutoForwardDiff`          | ✅    | ❌    | ✅     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoGTPSA`                | ✅    | ❌    | ❌     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoMooncake`             | ❌    | ✅    | ❌     | ❌      | ❌     | ❌      | ❌     | ❌      |
    | `AutoPolyesterForwardDiff` | 🔀    | ❌    | 🔀     | ✅      | ✅     | 🔀      | 🔀     | 🔀      |
    | `AutoReverseDiff`          | ❌    | 🔀    | ❌     | ✅      | ✅     | ✅      | ❌     | ❌      |
    | `AutoSymbolics`            | ✅    | ❌    | ✅     | ✅      | ✅     | ✅      | ✅     | ✅      |
    | `AutoTracker`              | ❌    | ✅    | ❌     | ✅      | ❌     | ❌      | ❌     | ❌      |
    | `AutoZygote`               | ❌    | ✅    | ❌     | ✅      | ✅     | ✅      | 🔀     | ❌      |

Moreover, each context type is supported by a specific subset of backends:

|                            | [`Constant`](@ref) | [`Cache`](@ref) |
| -------------------------- | ------------------ | --------------- |
| `AutoChainRules`           | ✅                  | ❌               |
| `AutoDiffractor`           | ❌                  | ❌               |
| `AutoEnzyme` (forward)     | ✅                  | ✅               |
| `AutoEnzyme` (reverse)     | ✅                  | ✅               |
| `AutoFastDifferentiation`  | ✅                  | ✅               |
| `AutoFiniteDiff`           | ✅                  | ✅               |
| `AutoFiniteDifferences`    | ✅                  | ✅               |
| `AutoForwardDiff`          | ✅                  | ✅               |
| `AutoGTPSA`                | ✅                  | ❌               |
| `AutoMooncake`             | ✅                  | ✅               |
| `AutoPolyesterForwardDiff` | ✅                  | ✅               |
| `AutoReverseDiff`          | ✅                  | ❌               |
| `AutoSymbolics`            | ✅                  | ❌               |
| `AutoTracker`              | ✅                  | ❌               |
| `AutoZygote`               | ✅                  | 🔀               |

## Second order

For second-order operators like [`second_derivative`](@ref), [`hessian`](@ref) and [`hvp`](@ref), there are two main options.
You can either use a single backend, or combine two of them within the [`SecondOrder`](@ref) struct:

```julia
backend = SecondOrder(outer_backend, inner_backend)
```

The inner backend will be called first, and the outer backend will differentiate the generated code.
In general, using a forward outer backend over a reverse inner backend will yield the best performance.

!!! danger
    Second-order AD is tricky, and many backend combinations will fail (even if you combine a backend with itself).
    Be ready to experiment and open issues if necessary.

## Backend switch

The wrapper [`DifferentiateWith`](@ref) allows you to switch between backends.
It takes a function `f` and specifies that `f` should be differentiated with the substitute backend of your choice, instead of whatever true backend the surrounding code is trying to use.
In other words, when someone tries to differentiate `dw = DifferentiateWith(f, substitute_backend)` with `true_backend`, then `substitute_backend` steps in and `true_backend` does not dive into the function `f` itself.
At the moment, `DifferentiateWith` only works when `true_backend` is either [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) or a [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible backend.

## Implementations

What follows is a list of implementation details from the package extensions of DifferentiationInterface.jl
It is not part of the public API or protected by semantic versioning, and it may become outdated.
When in doubt, refer to the code itself.

### ChainRulesCore

We only implement `pullback`, using the [`RuleConfig` mechanism](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html#config) to call back into AD.
Same-point preparation runs the forward sweep and returns the pullback closure.

### Diffractor

We only implement `pushforward`.

!!! danger
    The latest releases of Diffractor [broke DifferentiationInterface](https://github.com/JuliaDiff/Diffractor.jl/issues/290).

### Enzyme

Depending on the `mode` attribute inside [`AutoEnzyme`](@extref ADTypes.AutoEnzyme), we implement either `pushforward` or `pullback` based on `Enzyme.autodiff`.
When necessary, preparation chooses a number of chunks (for `gradient` and `jacobian` in forward mode, for `jacobian` only in reverse mode).

### FastDifferentiation

For every operator, preparation generates an [executable function](https://brianguenter.github.io/FastDifferentiation.jl/stable/makefunction/) from the symbolic expression of the differentiated function.

!!! warning
    Preparation can be very slow for symbolic AD.

### FiniteDiff

Whenever possible, preparation creates a cache object.
Pushforward is implemented rather slowly using a closure.

### FiniteDifferences

Nothing specific to mention.

### ForwardDiff

We implement [`pushforward`](@ref) directly using [`Dual` numbers](https://juliadiff.org/ForwardDiff.jl/stable/dev/how_it_works/), and preparation allocates the necessary space.
For higher level operators, preparation creates a [config object](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers), which can be type-unstable.

### GTPSA

For all operators, preparation preallocates the input [`TPS`s](https://bmad-sim.github.io/GTPSA.jl/stable/man/c_tps/), and for in-place functions the output `TPS`s as well. For minimal allocations of `TPS` temporaries inside of a function, the [`@FastGTPSA`/`@FastGTPSA!`](https://bmad-sim.github.io/GTPSA.jl/stable/man/j_fastgtpsa/) macros are recommended.

If a GTPSA [`Descriptor`](https://bmad-sim.github.io/GTPSA.jl/stable/man/b_descriptor/) is not provided to `AutoGTPSA`, then a `Descriptor` will be generated in preparation based on the context.

!!! danger
    When providing a custom GTPSA `Descriptor` to `AutoGTPSA`, it is the responsibility of the user to ensure that the number of [GTPSA "variables"](https://bmad-sim.github.io/GTPSA.jl/stable/quickstart/#Calculating-a-Truncated-Power-Series) specified in the `Descriptor` is consistent with the number of inputs of the provided function. Undefined behavior and crashes may occur if this is not the case.

### PolyesterForwardDiff

Most operators fall back on `AutoForwardDiff`.

### ReverseDiff

With `AutoReverseDiff(compile=false)`, preparation preallocates a [config](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractConfig-API).

With `AutoReverseDiff(compile=true)`, preparation records a [tape](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API) of the function's execution.
This tape is computed from the input `x` provided at preparation time.
It is control-flow dependent, so only one branch is recorded at each `if` statement.

!!! danger
    If your function has value-specific control flow (like `if x[1] > 0` or `if c == 1`), you may get silently wrong results whenever it takes new branches that were not taken during preparation.
    You must make sure to run preparation with an input and contexts whose values trigger the correct control flow for future executions.

Whenever contexts are provided, tape recording is deactivated in all cases, because otherwise the context values would be hardcoded into a tape.

### Symbolics

For all operators, preparation generates an [executable function](https://docs.sciml.ai/Symbolics/stable/manual/build_function/) from the symbolic expression of the differentiated function.

!!! warning
    Preparation can be very slow for symbolic AD.

### Mooncake

For `pullback`, preparation [builds the reverse rule](https://github.com/compintell/Mooncake.jl?tab=readme-ov-file#how-it-works) of the function.

### Tracker

We implement `pullback` based on `Tracker.back`.
Same-point preparation runs the forward sweep and returns the pullback closure at `x`.

### Zygote

We implement `pullback` based on `Zygote.pullback`.
Same-point preparation runs the forward sweep and returns the pullback closure at `x`.
