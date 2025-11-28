# Limitations

## Multithreading

The preparation result `prep` is not thread-safe, since it usually contains values that are mutated by differentiation calls.
Sharing it between threads may lead to unexpected behavior or errors.
If you need to run differentiation concurrently, construct a separate `prep` object for each thread, for instance with the help of [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl).

Note that functions which use multithreading internally are completely fine:

```julia
function f!(y, x)
    @threads for i in eachindex(y, x)
        y[i] = x[i]
    end
    return nothing
end

# this is correct
prep = prepare_jacobian(f!, y, backend, x)
J = jacobian(f!, y, prep, backend, x)
```

The pattern we are warning about concerns multithreading outside of the function:

```julia
# this is incorrect
prep = prepare_jacobian(f!, y, backend, x)
@threads for k in 1:n
    # same prep object, different threads writing to it
    J = jacobian(f!, ys[k], prep, backend, xs[k])
end
```

## Multiple active arguments

At the moment, most backends cannot work with multiple active (differentiated) arguments.
As a result, DifferentiationInterface only supports a single active argument, called `x` in the documentation.

## Complex numbers

Complex derivatives are only handled by a few AD backends, sometimes using different conventions.
To find the easiest common ground, DifferentiationInterface assumes that whenever complex numbers are involved, the function to differentiate is holomorphic.
This functionality is still considered experimental and not yet part of the public API guarantees.
If you work with non-holomorphic functions, you will need to manually separate real and imaginary parts.
