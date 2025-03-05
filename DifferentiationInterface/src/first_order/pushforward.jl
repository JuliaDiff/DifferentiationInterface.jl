## Docstrings

"""
    prepare_pushforward(f,     backend, x, tx, [contexts...]) -> prep
    prepare_pushforward(f!, y, backend, x, tx, [contexts...]) -> prep

Create a `prep` object that can be given to [`pushforward`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward end

"""
    prepare!_pushforward(f,     prep, backend, x, tx, [contexts...]) -> new_prep
    prepare!_pushforward(f!, y, prep, backend, x, tx, [contexts...]) -> new_prep

Same behavior as [`prepare_pushforward`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_pushforward end

"""
    prepare_pushforward_same_point(f,     backend, x, tx, [contexts...]) -> prep_same
    prepare_pushforward_same_point(f!, y, backend, x, tx, [contexts...]) -> prep_same

Create an `prep_same` object that can be given to [`pushforward`](@ref) and its variants _if they are applied at the same point `x` and with the same `contexts`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward_same_point end

"""
    value_and_pushforward(f,     [prep,] backend, x, tx, [contexts...]) -> (y, ty)
    value_and_pushforward(f!, y, [prep,] backend, x, tx, [contexts...]) -> (y, ty)

Compute the value and the pushforward of the function `f` at point `x` with a tuple of tangents `tx`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp`.

!!! info
    Required primitive for forward mode backends.
"""
function value_and_pushforward end

"""
    value_and_pushforward!(f,     dy, [prep,] backend, x, tx, [contexts...]) -> (y, ty)
    value_and_pushforward!(f!, y, dy, [prep,] backend, x, tx, [contexts...]) -> (y, ty)

Compute the value and the pushforward of the function `f` at point `x` with a tuple of tangents `tx`, overwriting `ty`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp!`.
"""
function value_and_pushforward! end

"""
    pushforward(f,     [prep,] backend, x, tx, [contexts...]) -> ty
    pushforward(f!, y, [prep,] backend, x, tx, [contexts...]) -> ty

Compute the pushforward of the function `f` at point `x` with a tuple of tangents `tx`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp`.
"""
function pushforward end

"""
    pushforward!(f,     dy, [prep,] backend, x, tx, [contexts...]) -> ty
    pushforward!(f!, y, dy, [prep,] backend, x, tx, [contexts...]) -> ty

Compute the pushforward of the function `f` at point `x` with a tuple of tangents `tx`, overwriting `ty`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp!`.
"""
function pushforward! end

## Preparation

struct PullbackPushforwardPrep{E} <: PushforwardPrep
    pullback_prep::E
end

function prepare_pushforward(
    f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_pushforward_aux(
        pushforward_performance(backend), f, backend, x, tx, contexts...
    )
end

function prepare_pushforward(
    f!::F, y, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_pushforward_aux(
        pushforward_performance(backend), f!, y, backend, x, tx, contexts...
    )
end

function _prepare_pushforward_aux(
    ::PushforwardSlow,
    f::F,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    dy = y isa Number ? one(y) : basis(y, first(CartesianIndices(y)))
    pullback_prep = prepare_pullback(f, backend, x, (dy,), contexts...)
    return PullbackPushforwardPrep(pullback_prep)
end

function _prepare_pushforward_aux(
    ::PushforwardSlow,
    f!::F,
    y,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    dy = y isa Number ? one(y) : basis(y, first(CartesianIndices(y)))
    pullback_prep = prepare_pullback(f!, y, backend, x, (dy,), contexts...)
    return PullbackPushforwardPrep(pullback_prep)
end

## One argument

function _pushforward_via_pullback(
    y::Number,
    f::F,
    pullback_prep::PullbackPrep,
    backend::AbstractADType,
    x,
    dx,
    contexts::Vararg{Context,C},
) where {F,C}
    a = only(pullback(f, pullback_prep, backend, x, (one(y),), contexts...))
    dy = dot(a, dx)
    return dy
end

function _pushforward_via_pullback(
    y::Complex,
    f::F,
    pullback_prep::PullbackPrep,
    backend::AbstractADType,
    x,
    dx,
    contexts::Vararg{Context,C},
) where {F,C}
    a = only(pullback(f, pullback_prep, backend, x, (one(y),), contexts...))
    b = only(pullback(f, pullback_prep, backend, x, (im * one(y),), contexts...))
    dy = real(dot(a, dx)) + im * real(dot(b, dx))
    return dy
end

function _pushforward_via_pullback(
    y::AbstractArray{<:Real},
    f::F,
    pullback_prep::PullbackPrep,
    backend::AbstractADType,
    x,
    dx,
    contexts::Vararg{Context,C},
) where {F,C}
    dy = map(CartesianIndices(y)) do i
        a = only(pullback(f, pullback_prep, backend, x, (basis(y, i),), contexts...))
        dot(a, dx)
    end
    return dy
end

function _pushforward_via_pullback(
    y::AbstractArray{<:Complex},
    f::F,
    pullback_prep::PullbackPrep,
    backend::AbstractADType,
    x,
    dx,
    contexts::Vararg{Context,C},
) where {F,C}
    dy = map(CartesianIndices(y)) do i
        a = only(pullback(f, pullback_prep, backend, x, (basis(y, i),), contexts...))
        b = only(pullback(f, pullback_prep, backend, x, (im * basis(y, i),), contexts...))
        real(dot(a, dx)) + im * real(dot(b, dx))
    end
    return dy
end

function value_and_pushforward(
    f::F,
    prep::PullbackPushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    (; pullback_prep) = prep
    y = f(x, map(unwrap, contexts)...)
    ty = ntuple(
        b -> _pushforward_via_pullback(y, f, pullback_prep, backend, x, tx[b], contexts...),
        Val(B),
    )
    return y, ty
end

function value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_ty = value_and_pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function pushforward(
    f::F,
    prep::PushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(f, prep, backend, x, tx, contexts...)[2]
end

function pushforward!(
    f::F,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(f, ty, prep, backend, x, tx, contexts...)[2]
end

## Two arguments

function _pushforward_via_pullback(
    f!::F,
    y::AbstractArray{<:Real},
    pullback_prep::PullbackPrep,
    backend::AbstractADType,
    x,
    dx,
    contexts::Vararg{Context,C},
) where {F,C}
    dy = map(CartesianIndices(y)) do i  # preserve shape
        a = only(pullback(f!, y, pullback_prep, backend, x, (basis(y, i),), contexts...))
        dot(a, dx)
    end
    return dy
end

function _pushforward_via_pullback(
    f!::F,
    y::AbstractArray{<:Complex},
    pullback_prep::PullbackPrep,
    backend::AbstractADType,
    x,
    dx,
    contexts::Vararg{Context,C},
) where {F,C}
    dy = map(CartesianIndices(y)) do i  # preserve shape
        a = only(pullback(f!, y, pullback_prep, backend, x, (basis(y, i),), contexts...))
        b = only(
            pullback(f!, y, pullback_prep, backend, x, (im * basis(y, i),), contexts...)
        )
        real(dot(a, dx)) + im * real(dot(b, dx))
    end
    return dy
end

function value_and_pushforward(
    f!::F,
    y,
    prep::PullbackPushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    (; pullback_prep) = prep
    ty = ntuple(
        b ->
            _pushforward_via_pullback(f!, y, pullback_prep, backend, x, tx[b], contexts...),
        Val(B),
    )
    f!(y, x, map(unwrap, contexts)...)
    return y, ty
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_ty = value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function pushforward(
    f!::F,
    y,
    prep::PushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)[2]
end

function pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AbstractADType,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)[2]
end

## Shuffled

function shuffled_single_pushforward(
    x,
    f::F,
    backend::AbstractADType,
    dx,
    rewrap::Rewrap{C},
    unannotated_contexts::Vararg{Any,C},
) where {F,C}
    ty = pushforward(f, backend, x, (dx,), rewrap(unannotated_contexts...)...)
    return only(ty)
end
