## Docstrings

"""
    prepare_pullback(f,     backend, x, ty, [contexts...]) -> prep
    prepare_pullback(f!, y, backend, x, ty, [contexts...]) -> prep

Create a `prep` object that can be given to [`pullback`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback end

"""
    prepare!_pullback(f,     prep, backend, x, ty, [contexts...]) -> new_prep
    prepare!_pullback(f!, y, prep, backend, x, ty, [contexts...]) -> new_prep

Same behavior as [`prepare_pullback`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_pullback end

"""
    prepare_pullback_same_point(f,     backend, x, ty, [contexts...]) -> prep_same
    prepare_pullback_same_point(f!, y, backend, x, ty, [contexts...]) -> prep_same

Create an `prep_same` object that can be given to [`pullback`](@ref) and its variants _if they are applied at the same point `x` and with the same `contexts`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback_same_point end

"""
    value_and_pullback(f,     [prep,] backend, x, ty, [contexts...]) -> (y, tx)
    value_and_pullback(f!, y, [prep,] backend, x, ty, [contexts...]) -> (y, tx)

Compute the value and the pullback of the function `f` at point `x` with a tuple of tangents `ty`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp`.

!!! info
    Required primitive for reverse mode backends.
"""
function value_and_pullback end

"""
    value_and_pullback!(f,     dx, [prep,] backend, x, ty, [contexts...]) -> (y, tx)
    value_and_pullback!(f!, y, dx, [prep,] backend, x, ty, [contexts...]) -> (y, tx)

Compute the value and the pullback of the function `f` at point `x` with a tuple of tangents `ty`, overwriting `dx`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp!`.
"""
function value_and_pullback! end

"""
    pullback(f,     [prep,] backend, x, ty, [contexts...]) -> tx
    pullback(f!, y, [prep,] backend, x, ty, [contexts...]) -> tx

Compute the pullback of the function `f` at point `x` with a tuple of tangents `ty`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp`.
"""
function pullback end

"""
    pullback!(f,     dx, [prep,] backend, x, ty, [contexts...]) -> tx
    pullback!(f!, y, dx, [prep,] backend, x, ty, [contexts...]) -> tx

Compute the pullback of the function `f` at point `x` with a tuple of tangents `ty`, overwriting `dx`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp!`.
"""
function pullback! end

## Preparation

struct PushforwardPullbackPrep{E} <: PullbackPrep
    pushforward_prep::E
end

function prepare_pullback(
    f::F, backend::AbstractADType, x, ty::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_pullback_aux(
        pullback_performance(backend), f, backend, x, ty, contexts...
    )
end

function prepare_pullback(
    f!::F, y, backend::AbstractADType, x, ty::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_pullback_aux(
        pullback_performance(backend), f!, y, backend, x, ty, contexts...
    )
end

function _prepare_pullback_aux(
    ::PullbackSlow,
    f::F,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = x isa Number ? one(x) : basis(x, first(CartesianIndices(x)))
    pushforward_prep = prepare_pushforward(f, backend, x, (dx,), contexts...)
    return PushforwardPullbackPrep(pushforward_prep)
end

function _prepare_pullback_aux(
    ::PullbackSlow,
    f!::F,
    y,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = x isa Number ? one(x) : basis(x, first(CartesianIndices(x)))
    pushforward_prep = prepare_pushforward(f!, y, backend, x, (dx,), contexts...)
    return PushforwardPullbackPrep(pushforward_prep)
end

## One argument

function _pullback_via_pushforward(
    f::F,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::Real,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    a = only(pushforward(f, pushforward_prep, backend, x, (one(x),), contexts...))
    dx = dot(a, dy)
    return dx
end

function _pullback_via_pushforward(
    f::F,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::Complex,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    a = only(pushforward(f, pushforward_prep, backend, x, (one(x),), contexts...))
    b = only(pushforward(f, pushforward_prep, backend, x, (im * one(x),), contexts...))
    dx = real(dot(a, dy)) + im * real(dot(b, dy))
    return dx
end

function _pullback_via_pushforward(
    f::F,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::AbstractArray{<:Real},
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = map(CartesianIndices(x)) do j
        a = only(pushforward(f, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        dot(a, dy)
    end
    return dx
end

function _pullback_via_pushforward(
    f::F,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::AbstractArray{<:Complex},
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = map(CartesianIndices(x)) do j
        a = only(pushforward(f, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        b = only(
            pushforward(f, pushforward_prep, backend, x, (im * basis(x, j),), contexts...),
        )
        real(dot(a, dy)) + im * real(dot(b, dy))
    end
    return dx
end

function value_and_pullback(
    f::F,
    prep::PushforwardPullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    (; pushforward_prep) = prep
    y = f(x, map(unwrap, contexts)...)
    tx = ntuple(
        b -> _pullback_via_pushforward(f, pushforward_prep, backend, x, ty[b], contexts...),
        Val(B),
    )
    return y, tx
end

function value_and_pullback!(
    f::F,
    tx::NTuple,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_tx = value_and_pullback(f, prep, backend, x, ty, contexts...)
    foreach(copyto!, tx, new_tx)
    return y, tx
end

function pullback(
    f::F,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function pullback!(
    f::F,
    tx::NTuple,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)[2]
end

## Two arguments

function _pullback_via_pushforward(
    f!::F,
    y,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::Real,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    a = only(pushforward(f!, y, pushforward_prep, backend, x, (one(x),), contexts...))
    dx = dot(a, dy)
    return dx
end

function _pullback_via_pushforward(
    f!::F,
    y,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::Complex,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    a = only(pushforward(f!, y, pushforward_prep, backend, x, (one(x),), contexts...))
    b = only(pushforward(f!, y, pushforward_prep, backend, x, (im * one(x),), contexts...))
    dx = real(dot(a, dy)) + im * real(dot(b, dy))
    return dx
end

function _pullback_via_pushforward(
    f!::F,
    y,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::AbstractArray{<:Real},
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = map(CartesianIndices(x)) do j  # preserve shape
        a = only(pushforward(f!, y, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        dot(a, dy)
    end
    return dx
end

function _pullback_via_pushforward(
    f!::F,
    y,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::AbstractArray{<:Complex},
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = map(CartesianIndices(x)) do j  # preserve shape
        a = only(pushforward(f!, y, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        b = only(
            pushforward(
                f!, y, pushforward_prep, backend, x, (im * basis(x, j),), contexts...
            ),
        )
        real(dot(a, dy)) + im * real(dot(b, dy))
    end
    return dx
end

function value_and_pullback(
    f!::F,
    y,
    prep::PushforwardPullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    (; pushforward_prep) = prep
    tx = ntuple(
        b -> _pullback_via_pushforward(
            f!, y, pushforward_prep, backend, x, ty[b], contexts...
        ),
        Val(B),
    )
    f!(y, x, map(unwrap, contexts)...)
    return y, tx
end

function value_and_pullback!(
    f!::F,
    y,
    tx::NTuple,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_tx = value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
    foreach(copyto!, tx, new_tx)
    return y, tx
end

function pullback(
    f!::F,
    y,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(f!, y, prep, backend, x, ty, contexts...)[2]
end

function pullback!(
    f!::F,
    y,
    tx::NTuple,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)[2]
end
