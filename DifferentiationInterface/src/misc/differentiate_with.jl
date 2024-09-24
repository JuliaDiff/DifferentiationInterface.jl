"""
    DifferentiateWith

Function wrapper that enforces differentiation with a "substitute" AD backend, possible different from the "true" AD backend that is called.

For instance, suppose a function `f` is not differentiable with Zygote because it involves mutation, but you know that it is differentiable with Enzyme.
Then `f2 = DifferentiateWith(f, AutoEnzyme())` is a new function that behaves like `f`, except that `f2` is differentiable with Zygote (thanks to a chain rule which calls Enzyme under the hood).
Moreover, any larger algorithm `alg` that calls `f2` instead of `f` will also be differentiable with Zygote (as long as `f` was the only Zygote blocker).

!!! tip
    This is mainly relevant for package developers who want to produce differentiable code at low cost, without writing the differentiation rules themselves.
    If you sprinkle a few `DifferentiateWith` in places where some AD backends may struggle, end users can pick from a wider variety of packages to differentiate your algorithms.

!!! warning
    `DifferentiateWith` only supports out-of-place functions `y = f(x)` without additional context arguments.
    It only makes these functions differentiable if the true backend is either [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) or compatible with [ChainRules](https://github.com/JuliaDiff/ChainRules.jl).
    For any other true backend, the differentiation behavior is not altered by `DifferentiateWith` (it becomes a transparent wrapper).

# Fields

- `f`: the function in question, with signature `f(x)`
- `backend::AbstractADType`: the substitute backend to use for differentiation

!!! note
    For the substitute AD backend to be called under the hood, its package needs to be loaded in addition to the package of the true AD backend.

# Constructor

    DifferentiateWith(f, backend)

# Example

```jldoctest
julia> using DifferentiationInterface

julia> import Enzyme, ForwardDiff, Zygote

julia> function f(x::Vector{Float64})
           a = Vector{Float64}(undef, 1)  # type constraint breaks ForwardDiff
           a[1] = sum(abs2, x)  # mutation breaks Zygote
           return a[1]
       end;

julia> f2 = DifferentiateWith(f, AutoEnzyme());

julia> f([3.0, 5.0]) == f2([3.0, 5.0])
true

julia> alg(x) = 7 * f2(x);

julia> ForwardDiff.gradient(alg, [3.0, 5.0])
2-element Vector{Float64}:
 42.0
 70.0

julia> Zygote.gradient(alg, [3.0, 5.0])[1]
2-element Vector{Float64}:
 42.0
 70.0
```
"""
struct DifferentiateWith{F,B<:AbstractADType}
    f::F
    backend::B
end

(dw::DifferentiateWith)(x) = dw.f(x)

function Base.show(io::IO, dw::DifferentiateWith)
    @compat (; f, backend) = dw
    return print(
        io,
        DifferentiateWith,
        "(",
        repr(f; context=io),
        ", ",
        repr(backend; context=io),
        ")",
    )
end
