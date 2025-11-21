## Docstrings

"""
    prepare_pullback(f,     backend, x, ty, [contexts...]; strict=Val(true)) -> prep
    prepare_pullback(f!, y, backend, x, ty, [contexts...]; strict=Val(true)) -> prep

$(docstring_prepare("pullback"; inplace = true))
"""
function prepare_pullback(
        f::F,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pullback_nokwarg(strict, f, backend, x, ty, contexts...)
end

function prepare_pullback(
        f!::F,
        y,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pullback_nokwarg(strict, f!, y, backend, x, ty, contexts...)
end

"""
    prepare!_pullback(f,     prep, backend, x, ty, [contexts...]) -> new_prep
    prepare!_pullback(f!, y, prep, backend, x, ty, [contexts...]) -> new_prep

$(docstring_prepare!("pullback"))
"""
function prepare!_pullback(
        f::F,
        old_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, old_prep, backend, x, ty, contexts...)
    return prepare_pullback_nokwarg(is_strict(old_prep), f, backend, x, ty, contexts...)
end

function prepare!_pullback(
        f!::F,
        y,
        old_prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, old_prep, backend, x, ty, contexts...)
    return prepare_pullback_nokwarg(is_strict(old_prep), f!, y, backend, x, ty, contexts...)
end

"""
    prepare_pullback_same_point(f,     backend, x, ty, [contexts...]; strict=Val(true)) -> prep_same
    prepare_pullback_same_point(f!, y, backend, x, ty, [contexts...]; strict=Val(true)) -> prep_same

$(docstring_prepare("pullback"; samepoint = true, inplace = true))
"""
function prepare_pullback_same_point(
        f::F,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pullback_same_point_nokwarg(strict, f, backend, x, ty, contexts...)
end

function prepare_pullback_same_point(
        f!::F,
        y,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C};
        strict::Val = Val(true),
    ) where {F, C}
    return prepare_pullback_same_point_nokwarg(strict, f!, y, backend, x, ty, contexts...)
end

function prepare_pullback_same_point_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, ty::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(strict, f, backend, x, ty, contexts...)
    return prepare_pullback_same_point(f, prep, backend, x, ty, contexts...)
end

function prepare_pullback_same_point_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(strict, f!, y, backend, x, ty, contexts...)
    return prepare_pullback_same_point(f!, y, prep, backend, x, ty, contexts...)
end

function prepare_pullback_same_point(
        f::F,
        prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, ty, contexts...)
    return prep
end

function prepare_pullback_same_point(
        f!::F,
        y,
        prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    return prep
end

"""
    value_and_pullback(f,     [prep,] backend, x, ty, [contexts...]) -> (y, tx)
    value_and_pullback(f!, y, [prep,] backend, x, ty, [contexts...]) -> (y, tx)

Compute the value and the pullback of the function `f` at point `x` with a tuple of tangents `ty`.

$(docstring_preparation_hint("pullback"; same_point = true))

!!! tip

    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp`.

!!! info

    Required primitive for reverse mode backends.
"""
function value_and_pullback(
        f::F, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f, backend, x, ty, contexts...)
    return value_and_pullback(f, prep, backend, x, ty, contexts...)
end

function value_and_pullback(
        f!::F, y, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f!, y, backend, x, ty, contexts...)
    return value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
end

"""
    value_and_pullback!(f,     dx, [prep,] backend, x, ty, [contexts...]) -> (y, tx)
    value_and_pullback!(f!, y, dx, [prep,] backend, x, ty, [contexts...]) -> (y, tx)

Compute the value and the pullback of the function `f` at point `x` with a tuple of tangents `ty`, overwriting `dx`.

$(docstring_preparation_hint("pullback"; same_point = true))

!!! tip

    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp!`.
"""
function value_and_pullback!(
        f::F, tx, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f, backend, x, ty, contexts...)
    return value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)
end

function value_and_pullback!(
        f!::F, y, tx, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f!, y, backend, x, ty, contexts...)
    return value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)
end

"""
    pullback(f,     [prep,] backend, x, ty, [contexts...]) -> tx
    pullback(f!, y, [prep,] backend, x, ty, [contexts...]) -> tx

Compute the pullback of the function `f` at point `x` with a tuple of tangents `ty`.

$(docstring_preparation_hint("pullback"; same_point = true))

!!! tip

    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp`.
"""
function pullback(
        f::F, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f, backend, x, ty, contexts...)
    return pullback(f, prep, backend, x, ty, contexts...)
end

function pullback(
        f!::F, y, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f!, y, backend, x, ty, contexts...)
    return pullback(f!, y, prep, backend, x, ty, contexts...)
end

"""
    pullback!(f,     dx, [prep,] backend, x, ty, [contexts...]) -> tx
    pullback!(f!, y, dx, [prep,] backend, x, ty, [contexts...]) -> tx

Compute the pullback of the function `f` at point `x` with a tuple of tangents `ty`, overwriting `dx`.

$(docstring_preparation_hint("pullback"; same_point = true))

!!! tip

    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp!`.
"""
function pullback!(
        f::F, tx, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f, backend, x, ty, contexts...)
    return pullback!(f, tx, prep, backend, x, ty, contexts...)
end

function pullback!(
        f!::F, y, tx, backend::AbstractADType, x, ty, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_pullback_nokwarg(Val(true), f!, y, backend, x, ty, contexts...)
    return pullback!(f!, y, tx, prep, backend, x, ty, contexts...)
end

## Preparation

struct PushforwardPullbackPrep{SIG, E} <: PullbackPrep{SIG}
    _sig::Val{SIG}
    pushforward_prep::E
end

function prepare_pullback_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, ty::NTuple, contexts::Vararg{Context, C}
    ) where {F, C}
    return _prepare_pullback_aux(
        strict, pullback_performance(backend), f, backend, x, ty, contexts...
    )
end

function prepare_pullback_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    return _prepare_pullback_aux(
        strict, pullback_performance(backend), f!, y, backend, x, ty, contexts...
    )
end

function _prepare_pullback_aux(
        strict::Val,
        ::PullbackSlow,
        f::F,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f, backend, x, ty, contexts...; strict)
    dx = if x isa Number
        oneunit(x)
    else
        basis(x)
    end
    pushforward_prep = prepare_pushforward_nokwarg(
        strict, f, backend, x, (dx,), contexts...
    )
    return PushforwardPullbackPrep(_sig, pushforward_prep)
end

function _prepare_pullback_aux(
        strict::Val,
        ::PullbackSlow,
        f!::F,
        y,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C}
    ) where {F, C}
    _sig = signature(f!, y, backend, x, ty, contexts...; strict)
    dx = if x isa Number
        oneunit(x)
    else
        basis(x)
    end
    pushforward_prep = prepare_pushforward_nokwarg(
        strict, f!, y, backend, x, (dx,), contexts...
    )
    return PushforwardPullbackPrep(_sig, pushforward_prep)
end

## One argument

function _value_and_pullback_via_pushforward(
        f::F,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::Real,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    y, a = onlysecond(value_and_pushforward(f, pushforward_prep, backend, x, (oneunit(x),), contexts...))
    tx = map(ty) do dy
        dot(a, dy)
    end
    return y, arroftup_to_tupofarr(tx)
end

function _value_and_pullback_via_pushforward(
        f::F,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::Complex,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    y, a = onlysecond(value_and_pushforward(f, pushforward_prep, backend, x, (oneunit(x),), contexts...))
    b = only(pushforward(f, pushforward_prep, backend, x, (im * oneunit(x),), contexts...))
    tx = map(ty) do dy
        real(dot(a, dy)) + im * real(dot(b, dy))
    end
    return y, arroftup_to_tupofarr(tx)
end

function _value_and_pullback_via_pushforward(
        f::F,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::AbstractArray{<:Real},
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    y = f(x, map(unwrap, contexts)...)
    tx = map(CartesianIndices(x)) do j
        a = only(pushforward(f, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        map(ty) do dy
            dot(a, dy)
        end
    end
    return y, arroftup_to_tupofarr(tx)
end

function _value_and_pullback_via_pushforward(
        f::F,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::AbstractArray{<:Complex},
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    y = f(x, map(unwrap, contexts)...)
    tx = map(CartesianIndices(x)) do j
        a = only(pushforward(f, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        b = only(
            pushforward(f, pushforward_prep, backend, x, (im * basis(x, j),), contexts...),
        )
        map(ty) do dy
            real(dot(a, dy)) + im * real(dot(b, dy))
        end
    end
    return y, arroftup_to_tupofarr(tx)
end

function value_and_pullback(
        f::F,
        prep::PushforwardPullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f, prep, backend, x, ty, contexts...)
    (; pushforward_prep) = prep
    return _value_and_pullback_via_pushforward(f, pushforward_prep, backend, x, ty, contexts...)
end

function value_and_pullback!(
        f::F,
        tx::NTuple,
        prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, ty, contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, ty, contexts...)
    return value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function pullback!(
        f::F,
        tx::NTuple,
        prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, ty, contexts...)
    return value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)[2]
end

## Two arguments

function _value_and_pullback_via_pushforward(
        f!::F,
        y,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::Real,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    _, a = onlysecond(value_and_pushforward(f!, y, pushforward_prep, backend, x, (oneunit(x),), contexts...))
    tx = map(ty) do dy
        dot(a, dy)
    end
    return y, arroftup_to_tupofarr(tx)
end

function _value_and_pullback_via_pushforward(
        f!::F,
        y,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::Complex,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    a = only(pushforward(f!, y, pushforward_prep, backend, x, (oneunit(x),), contexts...))
    _, b = onlysecond(
        value_and_pushforward(f!, y, pushforward_prep, backend, x, (im * oneunit(x),), contexts...)
    )
    tx = map(ty) do dy
        real(dot(a, dy)) + im * real(dot(b, dy))
    end
    return y, arroftup_to_tupofarr(tx)
end

function _value_and_pullback_via_pushforward(
        f!::F,
        y,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::AbstractArray{<:Real},
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    tx = map(CartesianIndices(x)) do j  # preserve shape
        _, a = onlysecond(value_and_pushforward(f!, y, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        map(ty) do dy
            dot(a, dy)
        end
    end
    return y, arroftup_to_tupofarr(tx)
end

function _value_and_pullback_via_pushforward(
        f!::F,
        y,
        pushforward_prep::PushforwardPrep,
        backend::AbstractADType,
        x::AbstractArray{<:Complex},
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    tx = map(CartesianIndices(x)) do j  # preserve shape
        a = only(pushforward(f!, y, pushforward_prep, backend, x, (basis(x, j),), contexts...))
        _, b = onlysecond(
            value_and_pushforward(
                f!, y, pushforward_prep, backend, x, (im * basis(x, j),), contexts...
            ),
        )
        map(ty) do dy
            real(dot(a, dy)) + im * real(dot(b, dy))
        end
    end
    return y, arroftup_to_tupofarr(tx)
end

function value_and_pullback(
        f!::F,
        y,
        prep::PushforwardPullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple{B},
        contexts::Vararg{Context, C},
    ) where {F, B, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; pushforward_prep) = prep
    return _value_and_pullback_via_pushforward(
        f!, y, pushforward_prep, backend, x, ty, contexts...
    )
end

function value_and_pullback!(
        f!::F,
        y,
        tx::NTuple,
        prep::PullbackPrep,
        backend::AbstractADType,
        x,
        ty::NTuple,
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f!, y, prep, backend, x, ty, contexts...)
    return value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)[2]
end
