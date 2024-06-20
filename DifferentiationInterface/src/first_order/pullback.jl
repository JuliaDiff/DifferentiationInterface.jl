## Docstrings

"""
    prepare_pullback(f,     backend, x, dy) -> extras
    prepare_pullback(f!, y, backend, x, dy) -> extras

Create an `extras` object that can be given to [`pullback`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback end

function prepare_pullback_batched end

"""
    prepare_pullback_same_point(f,     backend, x, dy) -> extras_same
    prepare_pullback_same_point(f!, y, backend, x, dy) -> extras_same

Create an `extras_same` object that can be given to [`pullback`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback_same_point end

function prepare_pullback_batched_same_point end

"""
    value_and_pullback(f,     backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback(f!, y, backend, x, dy, [extras]) -> (y, dx)

Compute the value and the pullback of the function `f` at point `x` with seed `dy`.

!!! info
    Required primitive for reverse mode backends.
"""
function value_and_pullback end

"""
    value_and_pullback!(f,     dx, backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)

Compute the value and the pullback of the function `f` at point `x` with seed `dy`, overwriting `dx`.
"""
function value_and_pullback! end

"""
    pullback(f,     backend, x, dy, [extras]) -> dx
    pullback(f!, y, backend, x, dy, [extras]) -> dx

Compute the pullback of the function `f` at point `x` with seed `dy`.
"""
function pullback end

function pullback_batched end

"""
    pullback!(f,     dx, backend, x, dy, [extras]) -> dx
    pullback!(f!, y, dx, backend, x, dy, [extras]) -> dx

Compute the pullback of the function `f` at point `x` with seed `dy`, overwriting `dx`.
"""
function pullback! end

function pullback_batched! end

## Preparation

### Extras types

"""
    PullbackExtras

Abstract type for additional information needed by [`pullback`](@ref) and its variants.
"""
abstract type PullbackExtras <: Extras end

struct NoPullbackExtras <: PullbackExtras end

struct PushforwardPullbackExtras{E} <: PullbackExtras
    pushforward_extras::E
end

## Standard

function prepare_pullback(f::F, backend::AbstractADType, x, dy) where {F}
    return prepare_pullback_aux(f, backend, x, dy, pullback_performance(backend))
end

function prepare_pullback(f!::F, y, backend::AbstractADType, x, dy) where {F}
    return prepare_pullback_aux(f!, y, backend, x, dy, pullback_performance(backend))
end

function prepare_pullback_aux(f::F, backend, x, dy, ::PullbackSlow) where {F}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f, backend, x, dx)
    return PushforwardPullbackExtras(pushforward_extras)
end

function prepare_pullback_aux(f!::F, y, backend, x, dy, ::PullbackSlow) where {F}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f!, y, backend, x, dx)
    return PushforwardPullbackExtras(pushforward_extras)
end

function prepare_pullback_aux(f, backend, x, dy, ::PullbackFast)
    throw(MissingBackendError(backend))
end

function prepare_pullback_aux(f!, y, backend, x, dy, ::PullbackFast)
    throw(MissingBackendError(backend))
end

### Standard, same point

function prepare_pullback_same_point(
    f::F, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return extras
end

function prepare_pullback_same_point(
    f!::F, y, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return extras
end

function prepare_pullback_same_point(f::F, backend::AbstractADType, x, dy) where {F}
    extras = prepare_pullback(f, backend, x, dy)
    return prepare_pullback_same_point(f, backend, x, dy, extras)
end

function prepare_pullback_same_point(f!::F, y, backend::AbstractADType, x, dy) where {F}
    extras = prepare_pullback(f!, y, backend, x, dy)
    return prepare_pullback_same_point(f!, y, backend, x, dy, extras)
end

### Batched

function prepare_pullback_batched(
    f::F, backend::AbstractADType, x, dy::Batch{B}
) where {F,B}
    return prepare_pullback(f, backend, x, first(dy.elements))
end

function prepare_pullback_batched(
    f!::F, y, backend::AbstractADType, x, dy::Batch{B}
) where {F,B}
    return prepare_pullback(f!, y, backend, x, first(dy.elements))
end

### Batched, same point

function prepare_pullback_batched_same_point(
    f::F, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    return prepare_pullback_same_point(f, backend, x, first(dy.elements), extras)
end

function prepare_pullback_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    return prepare_pullback_same_point(f!, y, backend, x, first(dy.elements), extras)
end

## One argument

### Standard

function value_and_pullback(f::F, backend::AbstractADType, x, dy) where {F}
    return value_and_pullback(f, backend, x, dy, prepare_pullback(f, backend, x, dy))
end

function value_and_pullback!(f::F, dx, backend::AbstractADType, x, dy) where {F}
    return value_and_pullback!(f, dx, backend, x, dy, prepare_pullback(f, backend, x, dy))
end

function pullback(f::F, backend::AbstractADType, x, dy) where {F}
    return pullback(f, backend, x, dy, prepare_pullback(f, backend, x, dy))
end

function pullback!(f::F, dx, backend::AbstractADType, x, dy) where {F}
    return pullback!(f, dx, backend, x, dy, prepare_pullback(f, backend, x, dy))
end

function value_and_pullback(
    f::F, backend, x, dy, extras::PushforwardPullbackExtras
) where {F}
    @compat (; pushforward_extras) = extras
    y = f(x)
    dx = if x isa Number && y isa Number
        dy * pushforward(f, backend, x, one(x), pushforward_extras)
    elseif x isa Number && y isa AbstractArray
        dot(dy, pushforward(f, backend, x, one(x), pushforward_extras))
    elseif x isa AbstractArray && y isa Number
        map(CartesianIndices(x)) do j
            dy * pushforward(f, backend, x, basis(backend, x, j), pushforward_extras)
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(dy, pushforward(f, backend, x, basis(backend, x, j), pushforward_extras))
        end
    end
    return y, dx
end

function value_and_pullback!(
    f::F, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    y, new_dx = value_and_pullback(f, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
end

function pullback(f::F, backend::AbstractADType, x, dy, extras::PullbackExtras) where {F}
    return value_and_pullback(f, backend, x, dy, extras)[2]
end

function pullback!(
    f::F, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

### Batched

function pullback_batched(
    f::F, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    dx_elements = ntuple(Val(B)) do l
        pullback(f, backend, x, dy.elements[l], extras)
    end
    return Batch(dx_elements)
end

function pullback_batched!(
    f::F, dx::Batch{B}, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    for l in 1:B
        pullback!(f, dx.elements[l], backend, x, dy.elements[l], extras)
    end
    return dx
end

## Two arguments

### Standard

function value_and_pullback(f!::F, y, backend::AbstractADType, x, dy) where {F}
    return value_and_pullback(
        f!, y, backend, x, dy, prepare_pullback(f!, y, backend, x, dy)
    )
end

function value_and_pullback!(f!::F, y, dx, backend::AbstractADType, x, dy) where {F}
    return value_and_pullback!(
        f!, y, dx, backend, x, dy, prepare_pullback(f!, y, backend, x, dy)
    )
end

function pullback(f!::F, y, backend::AbstractADType, x, dy) where {F}
    return pullback(f!, y, backend, x, dy, prepare_pullback(f!, y, backend, x, dy))
end

function pullback!(f!::F, y, dx, backend::AbstractADType, x, dy) where {F}
    return pullback!(f!, y, dx, backend, x, dy, prepare_pullback(f!, y, backend, x, dy))
end

function value_and_pullback(
    f!::F, y, backend, x, dy, extras::PushforwardPullbackExtras
) where {F}
    @compat (; pushforward_extras) = extras
    dx = if x isa Number && y isa AbstractArray
        dot(dy, pushforward(f!, y, backend, x, one(x), pushforward_extras))
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(dy, pushforward(f!, y, backend, x, basis(backend, x, j), pushforward_extras))
        end
    end
    f!(y, x)
    return y, dx
end

function value_and_pullback!(
    f!::F, y, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    y, new_dx = value_and_pullback(f!, y, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
end

function pullback(
    f!::F, y, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return value_and_pullback(f!, y, backend, x, dy, extras)[2]
end

function pullback!(
    f!::F, y, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return value_and_pullback!(f!, y, dx, backend, x, dy, extras)[2]
end

### Batched

function pullback_batched(
    f!::F, y, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    dx_elements = ntuple(Val(B)) do l
        pullback(f!, y, backend, x, dy.elements[l], extras)
    end
    return Batch(dx_elements)
end

function pullback_batched!(
    f!::F, y, dx::Batch{B}, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    for l in 1:B
        pullback!(f!, y, dx.elements[l], backend, x, dy.elements[l], extras)
    end
    return dx
end
