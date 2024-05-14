module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: NoGradientExtras, NoPullbackExtras, PullbackExtras
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient
using Compat

DI.check_available(::AutoTracker) = true
DI.twoarg_support(::AutoTracker) = DI.TwoArgNotSupported()

## Pullback

struct TrackerPullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoTracker, x, dy) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f, ::AutoTracker, x, dy, ::PullbackExtras=NoPullbackExtras()
)
    y, pb = forward(f, x)
    return TrackerPullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(f, ::AutoTracker, x, dy, ::NoPullbackExtras)
    y, pb = forward(f, x)
    return y, data(only(pb(dy)))
end

function DI.value_and_pullback(
    f, ::AutoTracker, x, dy, extras::TrackerPullbackExtrasSamePoint
)
    @compat (; y, pb) = extras
    return copy(y), data(only(pb(dy)))
end

function DI.pullback(f, ::AutoTracker, x, dy, extras::TrackerPullbackExtrasSamePoint)
    @compat (; pb) = extras
    return data(only(pb(dy)))
end

## Gradient

DI.prepare_gradient(f, ::AutoTracker, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::AutoTracker, x, ::NoGradientExtras)
    @compat (; val, grad) = withgradient(f, x)
    return val, data(only(grad))
end

function DI.gradient(f, ::AutoTracker, x, ::NoGradientExtras)
    @compat (; grad) = withgradient(f, x)
    return data(only(grad))
end

function DI.value_and_gradient!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    y, new_grad = DI.value_and_gradient(f, backend, x, extras)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    return copyto!(grad, DI.gradient(f, backend, x, extras))
end

end
