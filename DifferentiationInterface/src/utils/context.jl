"""
    Context

Abstract supertype for additional context arguments, which can be passed to differentiation operators after the active input `x` but are not differentiated.

# Subtypes

  - [`Constant`](@ref)
  - [`Cache`](@ref)
  - [`ConstantOrCache`](@ref)
"""
abstract type Context end

abstract type GeneralizedConstant <: Context end

unwrap(c::Context) = c.data
Base.:(==)(c1::Context, c2::Context) = unwrap(c1) == unwrap(c2)

## Public contexts

"""
    Constant

Concrete type of [`Context`](@ref) argument which is kept constant during differentiation.

Note that an operator can be prepared with an arbitrary value of the constant.
However, same-point preparation must occur with the exact value that will be reused later.

!!! warning

    Some backends require any `Constant` context to be a `Number` or an `AbstractArray`.

# Example

```jldoctest
julia> using DifferentiationInterface

julia> using ForwardDiff: ForwardDiff

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
adapt_eltype(c::Constant, ::Type) = c

"""
    Cache

Concrete type of [`Context`](@ref) argument which can be mutated with active values during differentiation.

The initial values present inside the cache do not matter.

For some backends, preparation allocates the required memory for `Cache` contexts with the right element type, similar to [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl).

!!! warning

    Some backends require any `Cache` context to be an `AbstractArray`, others accept nested (named) tuples of `AbstractArray`s.

# Example

```jldoctest
julia> using DifferentiationInterface

julia> using ForwardDiff: ForwardDiff

julia> f(x, c) = sum(copy!(c, x));

julia> prep = prepare_gradient(f, AutoForwardDiff(), [1.0, 2.0], Cache(zeros(2)));

julia> gradient(f, prep, AutoForwardDiff(), [3.0, 4.0], Cache(zeros(2)))
2-element Vector{Float64}:
 1.0
 1.0
```
"""
struct Cache{T} <: Context
    data::T
end

cache_maker(c) = Cache(c)
maker(::Cache) = cache_maker
adapt_eltype(c::Cache, ::Type{T}) where {T} = Cache(recursive_similar(unwrap(c), T))

"""
    ConstantOrCache

Concrete type of [`Context`](@ref) argument which can contain a mixture of constants and caches, passed along to the backend without modification.

Unlike for [`Cache`](@ref), it is up to the user to ensure that the internal storage can adapt to the required element types, for instance by using [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl) directly.
"""
struct ConstantOrCache{T} <: Context
    data::T
end

constantorcache_maker(c) = ConstantOrCache(c)
maker(::ConstantOrCache) = constantorcache_maker
adapt_eltype(c::ConstantOrCache, ::Type) = c

## Internal contexts for passing stuff around

"""
    FunctionContext

Private type of [`Context`](@ref) argument used for passing functions inside second-order differentiation.

Behaves differently for Enzyme only, where the function can be annotated.
"""
struct FunctionContext{T} <: GeneralizedConstant
    data::T
end

## Context manipulation

"""
    Rewrap

Utility for recording context types of additional arguments (e.g. `Constant` or `Cache`) and re-wrapping them into their types after they have been unwrapped.

Useful for second-order differentiation.
"""
struct Rewrap{C, T}
    context_makers::T
    function Rewrap(contexts::Vararg{Context, C}) where {C}
        context_makers = map(maker, contexts)
        return new{C, typeof(context_makers)}(context_makers)
    end
end

(::Rewrap{0})() = ()

function (r::Rewrap{C, T})(unannotated_contexts::Vararg{Any, C}) where {C, T}
    return map(r.context_makers, unannotated_contexts) do maker, c
        maker(c)
    end
end

## Closures

"""
    FixTail

Closure around a function `f` and a set of tail argument `tail_args` such that

```
(ft::FixTail)(args...) = ft.f(args..., ft.tail_args...)
```
"""
struct FixTail{F, A <: Tuple}
    f::F
    tail_args::A
    function FixTail(f::F, tail_args::Vararg{Any, N}) where {F, N}
        return new{F, typeof(tail_args)}(f, tail_args)
    end
end

function (ft::FixTail)(args::Vararg{Any, N}) where {N}
    return ft.f(args..., ft.tail_args...)
end

"""
    fix_tail(f, tail_args...)

Convenience for constructing a [`FixTail`](@ref), with a shortcut when there are no tail arguments.
"""
@inline fix_tail(f::F) where {F} = f
fix_tail(f::F, args::Vararg{Any, N}) where {F, N} = FixTail(f, args...)
