module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.check_available(::AutoFiniteDifferences) = true
DI.inplace_support(::AutoFiniteDifferences) = DI.InPlaceNotSupported()
DI.inner_preparation_behavior(::AutoFiniteDifferences) = DI.PrepareInnerSimple()

## Pushforward

function DI.prepare_pushforward(
    f, ::AutoFiniteDifferences, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoPushforwardPrep()
end

function DI.pushforward(
    f,
    ::DI.NoPushforwardPrep,
    backend::AutoFiniteDifferences,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    return f(x, map(DI.unwrap, contexts)...),
    DI.pushforward(f, prep, backend, x, tx, contexts...)
end

## Pullback

function DI.prepare_pullback(
    f, ::AutoFiniteDifferences, x, ty::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoPullbackPrep()
end

function DI.pullback(
    f,
    ::DI.NoPullbackPrep,
    backend::AutoFiniteDifferences,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    return f(x, map(DI.unwrap, contexts)...),
    DI.pullback(f, prep, backend, x, ty, contexts...)
end

## Gradient

function DI.prepare_gradient(
    f, ::AutoFiniteDifferences, x, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoGradientPrep()
end

function DI.gradient(
    f,
    ::DI.NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

## Jacobian

function DI.prepare_jacobian(
    f, ::AutoFiniteDifferences, x, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoJacobianPrep()
end

function DI.jacobian(
    f,
    ::DI.NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x, contexts...)
    return y, copyto!(jac, new_jac)
end

end
