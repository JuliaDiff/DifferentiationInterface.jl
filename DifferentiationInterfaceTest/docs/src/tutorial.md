# Tutorial

We present a typical workflow with DifferentiationInterfaceTest.jl, building on the tutorial of the [DifferentiationInterface.jl documentation](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface) (which we encourage you to read first).

```@repl tuto
using DifferentiationInterface, DifferentiationInterfaceTest
import ForwardDiff, Enzyme
```

## Introduction

The AD backends we want to compare are [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

```@example tuto
backends = [AutoForwardDiff(), AutoEnzyme(; mode=Enzyme.Reverse)]
```

To do that, we are going to take gradients of a simple function:

```@example tuto
f(x::AbstractArray) = sum(sin, x)
```

Of course we know the true gradient mapping:

```@example tuto
∇f(x::AbstractArray) = cos.(x)
```

DifferentiationInterfaceTest.jl relies with so-called "scenarios", in which you encapsulate the information needed for your test:

- the operator category (`:gradient`)
- the behavior of the operator (either `:in` or `:out` of place)
- the function `f`
- the input `x` of the function `f`
- the reference first-order result `res1` of the operator

```@example tuto
xv = rand(Float32, 3)
xm = rand(Float64, 3, 2)
scenarios = [
    Scenario{:gradient,:out}(f, xv; res1=∇f(xv)),
    Scenario{:gradient,:out}(f, xm; res1=∇f(xm))
];
nothing  # hide
```

## Testing

The main entry point for testing is the function [`test_differentiation`](@ref).
It has many options, but the main ingredients are the following:

```@repl tuto
test_differentiation(
    backends,  # the backends you want to compare
    scenarios,  # the scenarios you defined,
    correctness=true,  # compares values against the reference
    type_stability=false,  # checks type stability with JET.jl
    detailed=true,  # prints a detailed test set
)
```

If you are too lazy to manually specify the reference, you can also provide an AD backend as the `ref_backend` keyword argument, which will serve as the ground truth for comparison.

## Benchmarking

Once you are confident that your backends give the correct answers, you probably want to compare their performance.
This is made easy by the [`benchmark_differentiation`](@ref) function, whose syntax should feel familiar:

```@example tuto
df = benchmark_differentiation(backends, scenarios);
```

The resulting object is a `DataFrame` from [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl), whose columns correspond to the fields of [`DifferentiationBenchmarkDataRow`](@ref):
