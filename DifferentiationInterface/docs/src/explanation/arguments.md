# Arguments

## General guidelines

### Function form

DifferentiationInterface only computes derivatives for functions with one of two specific forms:

```julia
y = f(x, contexts...)  # out of place, returns `y`
f!(y, x, contexts...)  # in place, returns `nothing`
```

In this notation:

- `f` (or `f!`) is the differentiated function
- `y` is the output
- `x` is the input, the only "active" argument, which always comes first
- `contexts` may contain additional, inactive arguments

The quantities returned by the various [operators](@ref "Operators") always correspond to (partial) derivatives of `y` with respect to `x`.

### Assumptions

The package makes one central assumption on the behavior and implementation of `f` (or `f!`):

!!! danger "Mutation rule"
    Either an argument's provided value matters, or it can be mutated during the function call, but never both.

This rule is declined as follows:

- The provided value of `x` matters because we evaluate and differentiate `f` at point `x`. Therefore, `x` cannot be mutated by the function.
- For in-place functions `f!`, the output `y` is meant to be overwritten. Hence, its provided (initial) value cannot matter, and it must be entirely overwritten.

!!! warning
    Whether or not the function object itself can be mutated is a tricky question, and support for this varies between backends.
    When in doubt, try to avoid mutating functions and pass contexts instead.
    In any case, DifferentiationInterface will assume that the recursive components (fields, subfields, etc.) of `f` or `f!` individually satisfy the same mutation rule: whenever the initial value matters, no mutation is allowed.

## Contexts

### Motivation

As stated, there can be only one active argument, which we call `x`.
However, version 0.6 of the package introduced the possibility of additional "context" arguments, whose derivatives we don't need to compute.
Contexts can be useful if you have a function `y = f(x, a, b, c, ...)` or `f!(y, x, a, b, c, ...)` and you only want the derivative of `y` with respect to `x`.
Another option would be creating a closure, but that is sometimes undesirable for performance reasons.

Every context argument must be wrapped in a subtype of [`Context`](@ref) and come after the active argument `x`.

### Context types

There are three kinds of context: [`Constant`](@ref), [`Cache`](@ref) and the hybrid [`ConstantOrCache`](@ref).
Those are also classified based on the mutation rule:

- [`Constant`](@ref) contexts wrap data that influences the output of the function. Hence they cannot be mutated.
- [`Cache`](@ref) contexts correspond to scratch spaces that can be mutated at will. Hence their provided value is arbitrary.
- [`ConstantOrCache`](@ref) is a hybrid, whose recursive components (fields, subfields, etc.) must individually satisfy the assumptions of either `Constant` or `Cache`.

Semantically, both of these calls compute the partial gradient of `f(x, c)` with respect to `x`, but they consider `c` differently:

```julia
gradient(f, backend, x, Constant(c))
gradient(f, backend, x, Cache(c))
```

In the first call, `c` must be kept unchanged throughout the function evaluation.
In the second call, `c` may be mutated with values computed during the function.

!!! warning
    Not every backend supports every type of context. See the documentation on [backends](@ref "Backends") for more details.
