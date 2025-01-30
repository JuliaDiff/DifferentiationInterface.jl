module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoDiffractor
import DifferentiationInterface as DI
using Diffractor: DiffractorRuleConfig, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

DI.check_available(::AutoDiffractor) = true
DI.check_inplace(::AutoDiffractor) = false
DI.check_operator_overloading(::AutoDiffractor) = false
DI.pullback_performance(::AutoDiffractor) = DI.PullbackSlow()

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x, tx::NTuple) = DI.NoPushforwardPrep()

function DI.pushforward(f, ::DI.NoPushforwardPrep, ::AutoDiffractor, x, tx::NTuple)
    ty = map(tx) do dx
        # code copied from Diffractor.jl
        z = ∂☆{1}()(ZeroBundle{1}(f), bundle(x, dx))
        dy = z[TaylorTangentIndex(1)]
        dy
    end
    return ty
end

function DI.value_and_pushforward(
    f, prep::DI.NoPushforwardPrep, backend::AutoDiffractor, x, tx::NTuple
)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

end
