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
    prepare_pullback_same_point(f,     backend, x, ty, [contexts...]) -> prep_same
    prepare_pullback_same_point(f!, y, backend, x, ty, [contexts...]) -> prep_same

Create an `prep_same` object that can be given to [`pullback`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback_same_point end

"""
    value_and_pullback(f,     [prep,] backend, x, ty, [contexts...]) -> (y, tx)
    value_and_pullback(f!, y, [prep,] backend, x, ty, [contexts...]) -> (y, tx)

Compute the value and the pullback of the function `f` at point `x` with [`Tangents`](@ref) `ty`.

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

Compute the value and the pullback of the function `f` at point `x` with [`Tangents`](@ref) `ty`, overwriting `dx`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp!`.
"""
function value_and_pullback! end

"""
    pullback(f,     [prep,] backend, x, ty, [contexts...]) -> tx
    pullback(f!, y, [prep,] backend, x, ty, [contexts...]) -> tx

Compute the pullback of the function `f` at point `x` with [`Tangents`](@ref) `ty`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp`.
"""
function pullback end

"""
    pullback!(f,     dx, [prep,] backend, x, ty, [contexts...]) -> tx
    pullback!(f!, y, dx, [prep,] backend, x, ty, [contexts...]) -> tx

Compute the pullback of the function `f` at point `x` with [`Tangents`](@ref) `ty`, overwriting `dx`.

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
    f::F, backend::AbstractADType, x, ty::Tangents, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_pullback_aux(
        pullback_performance(backend), f, backend, x, ty, contexts...
    )
end

function prepare_pullback(
    f!::F, y, backend::AbstractADType, x, ty::Tangents, contexts::Vararg{Context,C}
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
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_prep = prepare_pushforward(f, backend, x, Tangents(dx), contexts...)
    return PushforwardPullbackPrep(pushforward_prep)
end

function _prepare_pullback_aux(
    ::PullbackSlow,
    f!::F,
    y,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_prep = prepare_pushforward(f!, y, backend, x, Tangents(dx), contexts...)
    return PushforwardPullbackPrep(pushforward_prep)
end

function _prepare_pullback_aux(
    ::PullbackFast, f, backend::AbstractADType, x, ty::Tangents, contexts::Vararg{Context}
)
    throw(MissingBackendError(backend))
end

function _prepare_pullback_aux(
    ::PullbackFast,
    f!,
    y,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context},
)
    throw(MissingBackendError(backend))
end

## One argument

function _pullback_via_pushforward(
    f::F,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::Number,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    t1 = pushforward(f, pushforward_prep, backend, x, Tangents(one(x)), contexts...)
    dx = dot(dy, only(t1))
    return dx
end

function _pullback_via_pushforward(
    f::F,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::AbstractArray,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = map(CartesianIndices(x)) do j
        t1 = pushforward(
            f, pushforward_prep, backend, x, Tangents(basis(backend, x, j)), contexts...
        )
        dot(dy, only(t1))
    end
    return dx
end

function value_and_pullback(
    f::F,
    prep::PushforwardPullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    @compat (; pushforward_prep) = prep
    y = f(x, map(unwrap, contexts)...)
    if B == 1
        dx = _pullback_via_pushforward(
            f, pushforward_prep, backend, x, only(ty), contexts...
        )
        return y, Tangents(dx)
    else
        dxs = ntuple(
            b -> _pullback_via_pushforward(
                f, pushforward_prep, backend, x, ty.d[b], contexts...
            ),
            Val(B),
        )
        return y, Tangents(dxs...)
    end
end

function value_and_pullback!(
    f::F,
    tx::Tangents,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_tx = value_and_pullback(f, prep, backend, x, ty, contexts...)
    return y, copyto!(tx, new_tx)
end

function pullback(
    f::F,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function pullback!(
    f::F,
    tx::Tangents,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents,
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
    x::Number,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    t1 = pushforward(f!, y, pushforward_prep, backend, x, Tangents(one(x)), contexts...)
    dx = dot(dy, only(t1))
    return dx
end

function _pullback_via_pushforward(
    f!::F,
    y,
    pushforward_prep::PushforwardPrep,
    backend::AbstractADType,
    x::AbstractArray,
    dy,
    contexts::Vararg{Context,C},
) where {F,C}
    dx = map(CartesianIndices(x)) do j  # preserve shape
        t1 = pushforward(
            f!,
            y,
            pushforward_prep,
            backend,
            x,
            Tangents(basis(backend, x, j)),
            contexts...,
        )
        dot(dy, only(t1))
    end
    return dx
end

function value_and_pullback(
    f!::F,
    y,
    prep::PushforwardPullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    @compat (; pushforward_prep) = prep
    if B == 1
        dx = _pullback_via_pushforward(
            f!, y, pushforward_prep, backend, x, only(ty), contexts...
        )
        f!(y, x, map(unwrap, contexts)...)
        return y, Tangents(dx)
    else
        dxs = ntuple(
            b -> _pullback_via_pushforward(
                f!, y, pushforward_prep, backend, x, ty.d[b], contexts...
            ),
            Val(B),
        )
        f!(y, x, map(unwrap, contexts)...)
        return y, Tangents(dxs...)
    end
end

function value_and_pullback!(
    f!::F,
    y,
    tx::Tangents,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_tx = value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
    return y, copyto!(tx, new_tx)
end

function pullback(
    f!::F,
    y,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback(f!, y, prep, backend, x, ty, contexts...)[2]
end

function pullback!(
    f!::F,
    y,
    tx::Tangents,
    prep::PullbackPrep,
    backend::AbstractADType,
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)[2]
end
