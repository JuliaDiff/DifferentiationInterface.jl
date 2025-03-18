module DifferentiationInterfaceZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Zygote:
    Buffer,
    ZygoteRuleConfig,
    gradient,
    hessian,
    jacobian,
    pullback,
    withgradient,
    withjacobian

DI.check_available(::AutoZygote) = true
DI.inplace_support(::AutoZygote) = DI.InPlaceNotSupported()

translate(c::DI.Context) = DI.unwrap(c)
translate(c::DI.Cache{<:AbstractArray}) = Buffer(DI.unwrap(c))
function translate(c::DI.Cache{<:Union{NTuple,NamedTuple}})
    return map(translate, map(DI.Cache, DI.unwrap(c)))
end

## Pullback

struct ZygotePullbackPrepSamePoint{SIG,Y,PB} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    y::Y
    pb::PB
end

function DI.prepare_pullback_nokwarg(
    strict::Val, f, backend::AutoZygote, x, ty::NTuple, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    return DI.NoPullbackPrep(_sig)
end

function DI.prepare_pullback_same_point(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    _sig = DI.signature(f, backend, x, ty, contexts...; strict=DI.is_strict(prep))
    y, pb = pullback(f, x, map(translate, contexts)...)
    return ZygotePullbackPrepSamePoint(_sig, y, pb)
end

function DI.value_and_pullback(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    y, pb = pullback(f, x, map(translate, contexts)...)
    tx = map(ty) do dy
        first(pb(dy))
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::ZygotePullbackPrepSamePoint,
    backend::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; y, pb) = prep
    tx = map(ty) do dy
        first(pb(dy))
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::ZygotePullbackPrepSamePoint,
    backend::AutoZygote,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; pb) = prep
    tx = map(ty) do dy
        first(pb(dy))
    end
    return tx
end

## Gradient

function DI.prepare_gradient_nokwarg(
    strict::Val, f, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return DI.NoGradientPrep(_sig)
end

function DI.value_and_gradient(
    f, prep::DI.NoGradientPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; val, grad) = withgradient(f, x, map(translate, contexts)...)
    return val, first(grad)
end

function DI.gradient(
    f, prep::DI.NoGradientPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    grad = gradient(f, x, map(translate, contexts)...)
    return first(grad)
end

function DI.value_and_gradient!(
    f, grad, prep::DI.NoGradientPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
    f, grad, prep::DI.NoGradientPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

## Jacobian

function DI.prepare_jacobian_nokwarg(
    strict::Val, f, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return DI.NoJacobianPrep(_sig)
end

function DI.value_and_jacobian(
    f, prep::DI.NoJacobianPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y = f(x, map(translate, contexts)...)
    # https://github.com/FluxML/Zygote.jl/issues/1506
    jac = jacobian(f, x, map(translate, contexts)...)
    return y, first(jac)
end

function DI.jacobian(
    f, prep::DI.NoJacobianPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    jac = jacobian(f, x, map(translate, contexts)...)
    return first(jac)
end

function DI.value_and_jacobian!(
    f, jac, prep::DI.NoJacobianPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x, contexts...)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(
    f, jac, prep::DI.NoJacobianPrep, backend::AutoZygote, x, contexts::Vararg{DI.Context,C}
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(jac, DI.jacobian(f, prep, backend, x, contexts...))
end

## HVP

# Beware, this uses ForwardDiff for the inner differentiation

struct ZygoteHVPPrep{SIG,P} <: DI.HVPPrep{SIG}
    _sig::Val{SIG}
    fd_prep::P
end

function DI.prepare_hvp_nokwarg(
    strict::Val, f, backend::AutoZygote, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    fd_prep = DI.prepare_hvp_nokwarg(
        strict, f, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
    return ZygoteHVPPrep(_sig, fd_prep)
end

function DI.hvp(
    f,
    prep::ZygoteHVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.hvp(
        f, prep.fd_prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::ZygoteHVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.hvp!(
        f, tg, prep.fd_prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp(
    f,
    prep::ZygoteHVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.gradient_and_hvp(
        f, prep.fd_prep, DI.SecondOrder(AutoForwardDiff(), backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp!(
    f,
    grad,
    tg::NTuple,
    prep::ZygoteHVPPrep,
    backend::AutoZygote,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.gradient_and_hvp!(
        f,
        grad,
        tg,
        prep.fd_prep,
        DI.SecondOrder(AutoForwardDiff(), backend),
        x,
        tx,
        contexts...,
    )
end

## Hessian

function DI.prepare_hessian_nokwarg(
    strict::Val, f, backend::AutoZygote, x, contexts::Vararg{DI.GeneralizedConstant,C}
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return DI.NoHessianPrep(_sig)
end

function DI.hessian(
    f,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.with_contexts(f, contexts...)
    hess = hessian(fc, x)
    return hess
end

function DI.hessian!(
    f,
    hess,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(hess, DI.hessian(f, prep, backend, x, contexts...))
end

function DI.value_gradient_and_hessian(
    f,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, grad = DI.value_and_gradient(f, backend, x, contexts...)
    hess = DI.hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::DI.NoHessianPrep,
    backend::AutoZygote,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, _ = DI.value_and_gradient!(f, grad, backend, x, contexts...)
    DI.hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end

end
