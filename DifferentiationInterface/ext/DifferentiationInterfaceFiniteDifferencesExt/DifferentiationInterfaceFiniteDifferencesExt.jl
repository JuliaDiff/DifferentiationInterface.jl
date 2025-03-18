module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.check_available(::AutoFiniteDifferences) = true
DI.inplace_support(::AutoFiniteDifferences) = DI.InPlaceNotSupported()
DI.inner_preparation_behavior(::AutoFiniteDifferences) = DI.PrepareInnerSimple()

## Pushforward

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f,
    backend::AutoFiniteDifferences,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    return DI.NoPushforwardPrep(_sig)
end

function DI.pushforward(
    f,
    prep::DI.NoPushforwardPrep,
    backend::AutoFiniteDifferences,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    fc = DI.with_contexts(f, contexts...)
    ty = map(tx) do dx
        jvp(backend.fdm, fc, (x, dx))
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::DI.NoPushforwardPrep,
    backend::AutoFiniteDifferences,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pushforward(f, prep, backend, x, tx, contexts...)
end

## Pullback

function DI.prepare_pullback_nokwarg(
    strict::Val,
    f,
    backend::AutoFiniteDifferences,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    return DI.NoPullbackPrep(_sig)
end

function DI.pullback(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoFiniteDifferences,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    fc = DI.with_contexts(f, contexts...)
    tx = map(ty) do dy
        only(j′vp(backend.fdm, fc, dy, x))
    end
    return tx
end

function DI.value_and_pullback(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoFiniteDifferences,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pullback(f, prep, backend, x, ty, contexts...)
end

## Gradient

function DI.prepare_gradient_nokwarg(
    strict::Val, f, backend::AutoFiniteDifferences, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return DI.NoGradientPrep(_sig)
end

function DI.gradient(
    f,
    prep::DI.NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.with_contexts(f, contexts...)
    return only(grad(backend.fdm, fc, x))
end

function DI.value_and_gradient(
    f,
    prep::DI.NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...), DI.gradient(f, prep, backend, x, contexts...)
end

function DI.gradient!(
    f,
    grad,
    prep::DI.NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::DI.NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

## Jacobian

function DI.prepare_jacobian_nokwarg(
    strict::Val, f, backend::AutoFiniteDifferences, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return DI.NoJacobianPrep(_sig)
end

function DI.jacobian(
    f,
    prep::DI.NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.with_contexts(f, contexts...)
    return only(jacobian(backend.fdm, fc, x))
end

function DI.value_and_jacobian(
    f,
    prep::DI.NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...), DI.jacobian(f, prep, backend, x, contexts...)
end

function DI.jacobian!(
    f,
    jac,
    prep::DI.NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(jac, DI.jacobian(f, prep, backend, x, contexts...))
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::DI.NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x, contexts...)
    return y, copyto!(jac, new_jac)
end

end
