"""
    Context

Abstract supertype for additional context arguments, which can be passed to differentiation operators after the active input `x` but are not differentiated.

# See also

- [`Constant`](@ref)
- [`Cache`](@ref)
"""
abstract type Context end

unwrap(c::Context) = c.data
Base.:(==)(c1::Context, c2::Context) = unwrap(c1) == unwrap(c2)

abstract type GeneralizedConstant <: Context end
abstract type GeneralizedCache <: Context end

## Public contexts

"""
    Constant

Concrete subtype of [`Context`](@ref) argument which is kept constant during differentiation.

Note that an operator can be prepared with an arbitrary value of the constant.
However, same-point preparation must occur with the exact value that will be reused later.

!!! warning
    Some backends require any `Constant` context to be a `Number` or an `AbstractArray`.

# Example

```jldoctest
julia> using DifferentiationInterface

julia> import ForwardDiff

julia> f(x, c) = c * sum(abs2, x);

julia> gradient(f, AutoForwardDiff(), [1.0, 2.0], Constant(10))
2-element Vector{Float64}:
 20.0
 40.0

julia> gradient(f, AutoForwardDiff(), [1.0, 2.0], Constant(100))
2-element Vector{Float64}:
 200.0
 400.0
```
"""
struct Constant{T} <: GeneralizedConstant
    data::T
end

constant_maker(c) = Constant(c)
maker(::Constant) = constant_maker

"""
    Cache

Concrete subtype of [`Context`](@ref) argument which can be mutated with active values during differentiation.

The initial values present inside the cache do not matter.

!!! warning
    Most backends require any `Cache` context to be an `AbstractArray`.
"""
struct Cache{T} <: GeneralizedCache
    data::T
end

cache_maker(c) = Cache(c)
maker(::Cache) = cache_maker

## Internal contexts for passing stuff around

"""
    FunctionContext

Concrete subtype of [`Context`](@ref) argument designed to contain functions, for internal use only.

It is mostly similar to [`Constant`](@ref).
"""
struct FunctionContext{T} <: GeneralizedConstant
    data::T
end

"""
    BackendContext

Concrete subtype of [`Context`](@ref) argument designed to contain backend objects, for internal use only.

It is mostly similar to [`Constant`](@ref).
"""
struct BackendContext{T<:AbstractADType} <: GeneralizedConstant
    data::T
end

"""
    PrepContext <: Context

Concrete subtype of [`Context`](@ref) argument designed to contain preparation objects, for internal use only.

It is mostly similar to [`Cache`](@ref).
"""
struct PrepContext{T<:Prep} <: GeneralizedCache
    data::T
end

"""
    UnknownContext <: Context

Concrete subtype of [`Context`](@ref) argument designed as a placeholder for when a future context value is not yet known, for internal use only.

It is relevant in second-order preparation.
"""
struct UnknownContext <: Context end

## Context manipulation

struct Rewrap{C,T}
    context_makers::T
    function Rewrap(contexts::Vararg{Context,C}) where {C}
        context_makers = map(maker, contexts)
        return new{C,typeof(context_makers)}(context_makers)
    end
end

(::Rewrap{0})() = ()

function (r::Rewrap{C,T})(unannotated_contexts::Vararg{Any,C}) where {C,T}
    return map(r.context_makers, unannotated_contexts) do maker, c
        maker(c)
    end
end

struct FixTail{F,A<:Tuple}
    f::F
    tail_args::A
end

function (ft::FixTail)(args::Vararg{Any,N}) where {N}
    return ft.f(args..., ft.tail_args...)
end

with_contexts(f) = f

function with_contexts(f::F, contexts::Vararg{Context,N}) where {F,N}
    tail_args = map(unwrap, contexts)
    return FixTail(f, tail_args)
end
