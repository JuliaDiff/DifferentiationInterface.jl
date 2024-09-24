module DifferentiationInterfaceZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
import DifferentiationInterface as DI
using DifferentiationInterface:
    Constant,
    HVPPrep,
    NoGradientPrep,
    NoHessianPrep,
    NoJacobianPrep,
    NoPullbackPrep,
    PullbackPrep,
    Tangents,
    unwrap
using ForwardDiff: ForwardDiff
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian
using Compat

DI.check_available(::AutoZygote) = true
DI.inplace_support(::AutoZygote) = DI.InPlaceNotSupported()

## Pullback

struct ZygotePullbackPrepSamePoint{Y,PB} <: PullbackPrep
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f, ::AutoZygote, x, ty::Tangents, contexts::Vararg{Constant,C}
) where {C}
    return NoPullbackPrep()
end

function DI.prepare_pullback_same_point(
    f, ::NoPullbackPrep, ::AutoZygote, x, ty::Tangents, contexts::Vararg{Constant,C}
) where {C}
    y, pb = pullback(f, x, map(unwrap, contexts)...)
    return ZygotePullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(
    f, ::NoPullbackPrep, ::AutoZygote, x, ty::Tangents, contexts::Vararg{Constant,C}
) where {C}
    y, pb = pullback(f, x, map(unwrap, contexts)...)
    tx = map(ty) do dy
        first(pb(dy))
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::ZygotePullbackPrepSamePoint,
    ::AutoZygote,
    x,
    ty::Tangents,
    contexts::Vararg{Constant,C},
) where {C}
    @compat (; y, pb) = prep
    tx = map(ty) do dy
        first(pb(dy))
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::ZygotePullbackPrepSamePoint,
    ::AutoZygote,
    x,
    ty::Tangents,
    contexts::Vararg{Constant,C},
) where {C}
    @compat (; pb) = prep
    tx = map(ty) do dy
        first(pb(dy))
    end
    return tx
end

## Gradient

function DI.prepare_gradient(f, ::AutoZygote, x, contexts::Vararg{Constant,C}) where {C}
    return NoGradientPrep()
end

function DI.value_and_gradient(
    f, ::NoGradientPrep, ::AutoZygote, x, contexts::Vararg{Constant,C}
) where {C}
    @compat (; val, grad) = withgradient(f, x, map(unwrap, contexts)...)
    return val, first(grad)
end

function DI.gradient(
    f, ::NoGradientPrep, ::AutoZygote, x, contexts::Vararg{Constant,C}
) where {C}
    return first(gradient(f, x, map(unwrap, contexts)...))
end

function DI.value_and_gradient!(
    f, grad, prep::NoGradientPrep, backend::AutoZygote, x, contexts::Vararg{Constant,C}
) where {C}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
    f, grad, prep::NoGradientPrep, backend::AutoZygote, x, contexts::Vararg{Constant,C}
) where {C}
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

## Jacobian

DI.prepare_jacobian(f, ::AutoZygote, x) = NoJacobianPrep()

function DI.value_and_jacobian(f, ::NoJacobianPrep, ::AutoZygote, x)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::NoJacobianPrep, ::AutoZygote, x)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!(f, jac, prep::NoJacobianPrep, backend::AutoZygote, x)
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(f, jac, prep::NoJacobianPrep, backend::AutoZygote, x)
    return copyto!(jac, DI.jacobian(f, prep, backend, x))
end

## HVP

# Beware, this uses ForwardDiff for the inner differentiation

struct ZygoteHVPPrep{G,PE} <: HVPPrep
    ∇f::G
    pushforward_prep::PE
end

function DI.prepare_hvp(f, ::AutoZygote, x, tx::Tangents)
    ∇f(x) = only(gradient(f, x))
    pushforward_prep = DI.prepare_pushforward(∇f, AutoForwardDiff(), x, tx)
    return ZygoteHVPPrep(∇f, pushforward_prep)
end

function DI.hvp(f, prep::ZygoteHVPPrep, ::AutoZygote, x, tx::Tangents)
    @compat (; ∇f, pushforward_prep) = prep
    return DI.pushforward(∇f, pushforward_prep, AutoForwardDiff(), x, tx)
end

function DI.hvp!(f, tg::Tangents, prep::ZygoteHVPPrep, ::AutoZygote, x, tx::Tangents)
    @compat (; ∇f, pushforward_prep) = prep
    return DI.pushforward!(∇f, tg, pushforward_prep, AutoForwardDiff(), x, tx)
end

## Hessian

DI.prepare_hessian(f, ::AutoZygote, x) = NoHessianPrep()

function DI.hessian(f, ::NoHessianPrep, ::AutoZygote, x)
    return hessian(f, x)
end

function DI.hessian!(f, hess, prep::NoHessianPrep, backend::AutoZygote, x)
    return copyto!(hess, DI.hessian(f, prep, backend, x))
end

function DI.value_gradient_and_hessian(f, prep::NoHessianPrep, backend::AutoZygote, x)
    y, grad = DI.value_and_gradient(f, NoGradientPrep(), backend, x)
    hess = DI.hessian(f, prep, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, prep::NoHessianPrep, backend::AutoZygote, x
)
    y, _ = DI.value_and_gradient!(f, grad, NoGradientPrep(), backend, x)
    DI.hessian!(f, hess, prep, backend, x)
    return y, grad, hess
end

end
