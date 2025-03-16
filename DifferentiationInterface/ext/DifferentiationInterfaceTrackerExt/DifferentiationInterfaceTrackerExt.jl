module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.check_available(::AutoTracker) = true
DI.inplace_support(::AutoTracker) = DI.InPlaceNotSupported()

## Pullback

struct TrackerPullbackPrepSamePoint{SIG,Y,PB} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f,
    backend::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    return DI.NoPullbackPrep(_sig)
end

function DI.prepare_pullback_same_point(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    _sig = DI.signature(f, prep, backend, x, ty, contexts...; strict)
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    y, pb = forward(f, x, map(DI.unwrap, contexts)...)
    return TrackerPullbackPrepSamePoint(_sig, y, pb)
end

function DI.value_and_pullback(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    y, pb = forward(f, x, map(DI.unwrap, contexts)...)
    tx = map(ty) do dy
        data(first(pb(dy)))
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::TrackerPullbackPrepSamePoint,
    backend::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; y, pb) = prep
    tx = map(ty) do dy
        data(first(pb(dy)))
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::TrackerPullbackPrepSamePoint,
    backend::AutoTracker,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; pb) = prep
    tx = map(ty) do dy
        data(first(pb(dy)))
    end
    return tx
end

## Gradient

function DI.prepare_gradient(
    f,
    backend::AutoTracker,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return DI.NoGradientPrep(_sig)
end

function DI.value_and_gradient(
    f,
    prep::DI.NoGradientPrep,
    backend::AutoTracker,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; val, grad) = withgradient(f, x, map(DI.unwrap, contexts)...)
    return val, data(first(grad))
end

function DI.gradient(
    f,
    prep::DI.NoGradientPrep,
    backend::AutoTracker,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; grad) = withgradient(f, x, map(DI.unwrap, contexts)...)
    return data(first(grad))
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::DI.NoGradientPrep,
    backend::AutoTracker,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
    f,
    grad,
    prep::DI.NoGradientPrep,
    backend::AutoTracker,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

end
