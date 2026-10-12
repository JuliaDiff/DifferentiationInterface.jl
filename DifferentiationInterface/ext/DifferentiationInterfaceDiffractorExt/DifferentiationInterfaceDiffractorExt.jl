module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoDiffractor
import DifferentiationInterface as DI
using Diffractor:
    Diffractor, AbstractTangentBundle, FwdMap, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

#=
Workaround for https://github.com/JuliaDiff/Diffractor.jl/issues/290 (type piracy).
The forward broadcast rule of Diffractor v0.2.7 to v0.2.10 calls `FwdMap{E}(f)` instead of `FwdMap{N,E}(f)`, which has no method.
=#
function Diffractor.FwdMap{E}(f::T) where {E, N, T <: AbstractTangentBundle{N}}
    return FwdMap{N, E, T}(f)
end

DI.check_available(::AutoDiffractor) = true
DI.inplace_support(::AutoDiffractor) = DI.InPlaceNotSupported()
DI.pullback_performance(::AutoDiffractor) = DI.PullbackSlow()

## Pushforward

function DI.prepare_pushforward_nokwarg(
        strict::Val, f, backend::AutoDiffractor, x, tx::NTuple
    )
    _sig = DI.signature(f, backend, x, tx; strict)
    return DI.NoPushforwardPrep(_sig)
end

function DI.pushforward(
        f, prep::DI.NoPushforwardPrep, backend::AutoDiffractor, x, tx::NTuple
    )
    DI.check_prep(f, prep, backend, x, tx)
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
    DI.check_prep(f, prep, backend, x, tx)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

end
