## Docstrings

"""
    prepare_pushforward(f,     backend, x, tx, [contexts...]; strict=Val(true)) -> prep
    prepare_pushforward(f!, y, backend, x, tx, [contexts...]; strict=Val(true)) -> prep

$(docstring_prepare("pushforward"; inplace = true))
"""
function prepare_pushforward(
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pushforward_nokwarg(strict, f, backend, x, tx, contexts...)
end

function prepare_pushforward(
        f!::F,
        y,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pushforward_nokwarg(strict, f!, y, backend, x, tx, contexts...)
end

"""
    prepare!_pushforward(f,     prep, backend, x, tx, [contexts...]) -> new_prep
    prepare!_pushforward(f!, y, prep, backend, x, tx, [contexts...]) -> new_prep

$(docstring_prepare!("pushforward"))
"""
function prepare!_pushforward(
        f::F,
        old_prep::PushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, old_prep, backend, x, tx, contexts...)
    return prepare_pushforward_nokwarg(is_strict(old_prep), f, backend, x, tx, contexts...)
end

function prepare!_pushforward(
        f!::F,
        y,
        old_prep::PushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, old_prep, backend, x, tx, contexts...)
    return prepare_pushforward_nokwarg(
        is_strict(old_prep), f!, y, backend, x, tx, contexts...
    )
end

"""
    prepare_pushforward_same_point(f,     backend, x, tx, [contexts...]; strict=Val(true)) -> prep_same
    prepare_pushforward_same_point(f!, y, backend, x, tx, [contexts...]; strict=Val(true)) -> prep_same

$(docstring_prepare("pushforward"; samepoint = true, inplace = true))
"""
function prepare_pushforward_same_point(
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pushforward_same_point_nokwarg(strict, f, backend, x, tx, contexts...)
end

function prepare_pushforward_same_point(
        f!::F,
        y,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pushforward_same_point_nokwarg(
        strict, f!, y, backend, x, tx, contexts...
    )
end

function prepare_pushforward_same_point_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(strict, f, backend, x, tx, contexts...)
    return prepare_pushforward_same_point(f, prep, backend, x, tx, contexts...)
end

function prepare_pushforward_same_point_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(strict, f!, y, backend, x, tx, contexts...)
    return prepare_pushforward_same_point(f!, y, prep, backend, x, tx, contexts...)
end

function prepare_pushforward_same_point(
        f::F,
        prep::PushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    return prep
end

function prepare_pushforward_same_point(
        f!::F,
        y,
        prep::PushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    return prep
end

"""
    value_and_pushforward(f,     [prep,] backend, x, tx, [contexts...]) -> (y, ty)
    value_and_pushforward(f!, y, [prep,] backend, x, tx, [contexts...]) -> (y, ty)

Compute the value and the pushforward of the function `f` at point `x` with a tuple of tangents `tx`.

$(docstring_preparation_hint("pushforward"; same_point = true))

!!! tip

    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp`.

!!! info

    Required primitive for forward mode backends.
"""
function value_and_pushforward(
        f::F, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return value_and_pushforward(f, prep, backend, x, tx, contexts...)
end

function value_and_pushforward(
        f!::F, y, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f!, y, backend, x, tx, contexts...)
    return value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
end

"""
    value_and_pushforward!(f,     dy, [prep,] backend, x, tx, [contexts...]) -> (y, ty)
    value_and_pushforward!(f!, y, dy, [prep,] backend, x, tx, [contexts...]) -> (y, ty)

Compute the value and the pushforward of the function `f` at point `x` with a tuple of tangents `tx`, overwriting `ty`.

$(docstring_preparation_hint("pushforward"; same_point = true))

!!! tip

    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp!`.
"""
function value_and_pushforward!(
        f::F, ty, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return value_and_pushforward!(f, ty, prep, backend, x, tx, contexts...)
end

function value_and_pushforward!(
        f!::F, y, ty, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f!, y, backend, x, tx, contexts...)
    return value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
end

"""
    pushforward(f,     [prep,] backend, x, tx, [contexts...]) -> ty
    pushforward(f!, y, [prep,] backend, x, tx, [contexts...]) -> ty

Compute the pushforward of the function `f` at point `x` with a tuple of tangents `tx`.

$(docstring_preparation_hint("pushforward"; same_point = true))

!!! tip

    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp`.
"""
function pushforward(
        f::F, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return pushforward(f, prep, backend, x, tx, contexts...)
end

function pushforward(
        f!::F, y, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f!, y, backend, x, tx, contexts...)
    return pushforward(f!, y, prep, backend, x, tx, contexts...)
end

"""
    pushforward!(f,     dy, [prep,] backend, x, tx, [contexts...]) -> ty
    pushforward!(f!, y, dy, [prep,] backend, x, tx, [contexts...]) -> ty

Compute the pushforward of the function `f` at point `x` with a tuple of tangents `tx`, overwriting `ty`.

$(docstring_preparation_hint("pushforward"; same_point = true))

!!! tip

    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp!`.
"""
function pushforward!(
        f::F, ty, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f, backend, x, tx, contexts...)
    return pushforward!(f, ty, prep, backend, x, tx, contexts...)
end

function pushforward!(
        f!::F, y, ty, backend::AbstractADType, x, tx, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pushforward_nokwarg(Val(true), f!, y, backend, x, tx, contexts...)
    return pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
end

## Preparation

struct PullbackPushforwardPrep{SIG, E, Y} <: PushforwardPrep{SIG}
    _sig::Val{SIG}
    pullback_prep::E
    y_example::Y
end

function prepare_pushforward_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, tx::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    return _prepare_pushforward_aux(
        strict, pushforward_performance(backend), f, backend, x, tx, contexts...
    )
end

function prepare_pushforward_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    return _prepare_pushforward_aux(
        strict, pushforward_performance(backend), f!, y, backend, x, tx, contexts...
    )
end

function _prepare_pushforward_aux(
        strict::Val,
        ::PushforwardSlow,
        f::F,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, tx, contexts...; strict)
    y = f(x, map(unwrap, contexts)...)
    dy = if y isa Number
        oneunit(y)
    else
        basis(y)
    end
    pullback_prep = prepare_pullback_nokwarg(strict, f, backend, x, (dy,), contexts...)
    return PullbackPushforwardPrep(_sig, pullback_prep, y)
end

function _prepare_pushforward_aux(
        strict::Val,
        ::PushforwardSlow,
        f!::F,
        y,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f!, y, backend, x, tx, contexts...; strict)
    dy = basis(y)
    pullback_prep = prepare_pullback_nokwarg(strict, f!, y, backend, x, (dy,), contexts...)
    return PullbackPushforwardPrep(_sig, pullback_prep, y)
end

## One argument

function _value_and_pushforward_via_pullback(
        y_ex::Number,
        f::F,
        pullback_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        dx,
        contexts::Vararg{Context, C},
    ) where {F, C}
    y, a = onlysecond(value_and_pullback(f, pullback_prep, backend, x, (oneunit(y_ex),), contexts...))
    dy = dot(a, dx)
    return y, dy
end

function _value_and_pushforward_via_pullback(
        y_ex::Complex,
        f::F,
        pullback_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        dx,
        contexts::Vararg{Context, C},
    ) where {F, C}
    y, a = onlysecond(value_and_pullback(f, pullback_prep, backend, x, (oneunit(y_ex),), contexts...))
    b = only(pullback(f, pullback_prep, backend, x, (im * oneunit(y_ex),), contexts...))
    dy = real(dot(a, dx)) + im * real(dot(b, dx))
    return y, dy
end

function _value_and_pushforward_via_pullback(
        y_ex::AbstractArray{<:Real},
        f::F,
        pullback_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        dx,
        contexts::Vararg{Context, C},
    ) where {F, C}
    y = f(x, map(unwrap, contexts)...)
    dy = map(CartesianIndices(y_ex)) do i
        a = only(pullback(f, pullback_prep, backend, x, (basis(y_ex, i),), contexts...))
        dot(a, dx)
    end
    return y, dy
end

function _value_and_pushforward_via_pullback(
        y_ex::AbstractArray{<:Complex},
        f::F,
        pullback_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        dx,
        contexts::Vararg{Context, C},
    ) where {F, C}
    y = f(x, map(unwrap, contexts)...)
    dy = map(CartesianIndices(y_ex)) do i
        a = only(pullback(f, pullback_prep, backend, x, (basis(y_ex, i),), contexts...))
        b = only(pullback(f, pullback_prep, backend, x, (im * basis(y_ex, i),), contexts...))
        real(dot(a, dx)) + im * real(dot(b, dx))
    end
    return y, dy
end

function value_and_pushforward(
        f::F,
        prep::PullbackPushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    (; pullback_prep, y_example) = prep
    ys_and_ty = ntuple(
        b -> _value_and_pushforward_via_pullback(y_example, f, pullback_prep, backend, x, tx[b], contexts...),
        Val(B),
    )
    y = first(first(ys_and_ty))
    ty = map(last, ys_and_ty)
    return y, ty
end

function value_and_pushforward!(
        f::F,
        ty::NTuple,
        prep::PushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    return value_and_pushforward(f, prep, backend, x, tx, contexts...)[2]
end

function pushforward!(
        f::F,
        ty::NTuple,
        prep::PushforwardPrep,
        backend::AbstractADType,
        x,
        tx::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, tx, contexts...)
    return value_and_pushforward!(f, ty, prep, backend, x, tx, contexts...)[2]
end

## Two arguments

function _value_and_pushforward_via_pullback(
        f!::F,
        y::AbstractArray{<:Real},
        pullback_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        dx,
        contexts::Vararg{Context, C},
    ) where {F, C}
    dy = map(CartesianIndices(y)) do i  # preserve shape
        _, a = onlysecond(value_and_pullback(f!, y, pullback_prep, backend, x, (basis(y, i),), contexts...))
        dot(a, dx)
    end
    return dy
end

function _value_and_pushforward_via_pullback(
        f!::F,
        y::AbstractArray{<:Complex},
        pullback_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        dx,
        contexts::Vararg{Context, C},
    ) where {F, C}
    dy = map(CartesianIndices(y)) do i  # preserve shape
        a = only(pullback(f!, y, pullback_prep, backend, x, (basis(y, i),), contexts...))
        _, b = onlysecond(
            value_and_pullback(f!, y, pullback_prep, backend, x, (im * basis(y, i),), contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; pullback_prep) = prep
    ty = ntuple(
        b ->
        _value_and_pushforward_via_pullback(f!, y, pullback_prep, backend, x, tx[b], contexts...),
        Val(B),
    )
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, tx, contexts...)
    return value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)[2]
end

## Shuffled

function shuffled_single_pushforward(
        x,
        f::F,
        backend::AbstractADType,
        dx,
        rewrap::Rewrap{C},
        unannotated_contexts::Vararg{Any, C},
    ) where {F, C}
    ty = pushforward(f, backend, x, (dx,), rewrap(unannotated_contexts...)...)
    return only(ty)
end
