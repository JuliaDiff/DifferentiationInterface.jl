module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: NoGradientExtras, NoPullbackExtras
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.supports_mutation(::AutoTracker) = DI.MutationNotSupported()

## Pullback

DI.prepare_pullback(f, ::AutoTracker, x) = NoPullbackExtras()

function DI.value_and_pullback_split(f, ::AutoTracker, x, ::NoPullbackExtras)
    y, back = forward(f, x)
    pullbackfunc(dy) = data(only(back(dy)))
    return y, pullbackfunc
end

## Gradient

DI.prepare_gradient(f, ::AutoTracker, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::AutoTracker, x, ::NoGradientExtras)
    (; val, grad) = withgradient(f, x)
    return val, data(only(grad))
end

function DI.gradient(f, ::AutoTracker, x, ::NoGradientExtras)
    (; grad) = withgradient(f, x)
    return data(only(grad))
end

function DI.value_and_gradient!!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AutoTracker, x, extras::NoGradientExtras)
    return DI.gradient(f, backend, x, extras)
end

end
