module DifferentiationInterfaceHyperHessiansExt

import DifferentiationInterface as DI
import .DI: AutoHyperHessians
using ADTypes: ForwardMode
using HyperHessians:
    HVPConfig,
    HessianConfig,
    Chunk,
    chunksize,
    pickchunksize,
    HyperDual,
    hessian,
    hessian!,
    hessian_gradient_value,
    hessian_gradient_value!,
    hessian,
    hvp,
    hvp!,
    hvp_gradient_value,
    hvp_gradient_value!

## Traits
DI.check_available(::DI.AutoHyperHessians) = true
DI.inplace_support(::DI.AutoHyperHessians) = DI.InPlaceSupported()
DI.mode(::DI.AutoHyperHessians) = ForwardMode()

chunk_from_backend(backend::DI.AutoHyperHessians, x) =
    isnothing(backend.chunksize) ? Chunk(x) : Chunk{backend.chunksize}()
chunk_from_backend(backend::DI.AutoHyperHessians, N::Integer, ::Type{T}) where {T} =
    isnothing(backend.chunksize) ? Chunk(pickchunksize(N, T), T) : Chunk{backend.chunksize}()

function DI.pick_batchsize(backend::DI.AutoHyperHessians, x::AbstractArray)
    B = chunksize(chunk_from_backend(backend, x))
    return DI.BatchSizeSettings{B}(length(x))
end

function DI.pick_batchsize(backend::DI.AutoHyperHessians, N::Integer)
    B = chunksize(chunk_from_backend(backend, N, Float64))
    return DI.BatchSizeSettings{B}(N)
end

function DI.threshold_batchsize(backend::DI.AutoHyperHessians, chunksize2::Integer)
    chunksize1 = backend.chunksize
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return DI.AutoHyperHessians(; chunksize)
end

function _translate_toprep(::Type{T}, c::Union{DI.GeneralizedConstant, DI.ConstantOrCache}) where {T}
    return nothing
end
function _translate_toprep(::Type{T}, c::DI.Cache) where {T}
    return DI.recursive_similar(DI.unwrap(c), T)
end

function translate_toprep(::Type{T}, contexts::NTuple{C, DI.Context}) where {T, C}
    new_contexts = map(contexts) do c
        _translate_toprep(T, c)
    end
    return new_contexts
end

function _translate_prepared(c::Union{DI.GeneralizedConstant, DI.ConstantOrCache}, _pc)
    return DI.unwrap(c)
end
_translate_prepared(_c::DI.Cache, pc) = pc

function translate_prepared(
        contexts::NTuple{C, DI.Context}, prep_contexts::NTuple{C, Any}
    ) where {C}
    new_contexts = map(contexts, prep_contexts) do c, pc
        _translate_prepared(c, pc)
    end
    return new_contexts
end

## Second derivative (scalar input)

struct HyperHessiansSecondDerivativePrep{SIG, C} <: DI.SecondDerivativePrep{SIG}
    _sig::Val{SIG}
    contexts_prepared::C
end

function DI.prepare_second_derivative_nokwarg(
        strict::Val, f, backend::DI.AutoHyperHessians, x::Number, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    contexts_prepared = translate_toprep(HyperDual{1, 1, typeof(x)}, contexts)
    return HyperHessiansSecondDerivativePrep(_sig, contexts_prepared)
end

function DI.second_derivative(
        f,
        prep::HyperHessiansSecondDerivativePrep,
        backend::DI.AutoHyperHessians,
        x::Number,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    return hessian(fc, x)
end

function DI.second_derivative!(
        f,
        der2,
        prep::HyperHessiansSecondDerivativePrep,
        backend::DI.AutoHyperHessians,
        x::Number,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    copyto!(der2, DI.second_derivative(f, prep, backend, x, contexts...))
    return der2
end

function DI.value_derivative_and_second_derivative(
        f,
        prep::HyperHessiansSecondDerivativePrep,
        backend::DI.AutoHyperHessians,
        x::Number,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    res = hessian_gradient_value(fc, x)
    return res.value, res.gradient, res.hessian
end

function DI.value_derivative_and_second_derivative!(
        f,
        der,
        der2,
        prep::HyperHessiansSecondDerivativePrep,
        backend::DI.AutoHyperHessians,
        x::Number,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_der, new_der2 = DI.value_derivative_and_second_derivative(f, prep, backend, x, contexts...)
    copyto!(der, new_der)
    copyto!(der2, new_der2)
    return y, der, der2
end

## Preparation structs

struct HyperHessiansHessianPrep{SIG, C, CP} <: DI.HessianPrep{SIG}
    _sig::Val{SIG}
    cfg::C
    contexts_prepared::CP
end

struct HyperHessiansHVPPrep{SIG, C, CP} <: DI.HVPPrep{SIG}
    _sig::Val{SIG}
    cfg::C
    contexts_prepared::CP
end

## Hessian

function DI.prepare_hessian_nokwarg(
        strict::Val, f, backend::DI.AutoHyperHessians, x::AbstractArray, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    cfg = HessianConfig(x, chunk_from_backend(backend, x))
    contexts_prepared = translate_toprep(eltype(cfg.duals), contexts)
    return HyperHessiansHessianPrep(_sig, cfg, contexts_prepared)
end

function DI.hessian(
        f,
        prep::HyperHessiansHessianPrep,
        backend::DI.AutoHyperHessians,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    return hessian(fc, x, prep.cfg)
end

function DI.hessian!(
        f,
        hess,
        prep::HyperHessiansHessianPrep,
        backend::DI.AutoHyperHessians,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    return hessian!(hess, fc, x, prep.cfg)
end

function DI.value_gradient_and_hessian(
        f,
        prep::HyperHessiansHessianPrep,
        backend::DI.AutoHyperHessians,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    res = hessian_gradient_value(fc, x, prep.cfg)
    return res.value, res.gradient, res.hessian
end

function DI.value_gradient_and_hessian!(
        f,
        grad,
        hess,
        prep::HyperHessiansHessianPrep,
        backend::DI.AutoHyperHessians,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    val = hessian_gradient_value!(hess, grad, fc, x, prep.cfg)
    return val, grad, hess
end

## HVP

function DI.prepare_hvp_nokwarg(
        strict::Val, f, backend::DI.AutoHyperHessians, x::AbstractArray, tx::NTuple, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    cfg = HVPConfig(x, tx, chunk_from_backend(backend, x))
    contexts_prepared = translate_toprep(eltype(cfg.duals), contexts)
    return HyperHessiansHVPPrep(_sig, cfg, contexts_prepared)
end

function DI.prepare_hvp_same_point(
        f,
        prep::HyperHessiansHVPPrep,
        backend::DI.AutoHyperHessians,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return prep
end

function DI.hvp(
        f,
        prep::HyperHessiansHVPPrep,
        backend::AutoHyperHessians,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    return hvp(fc, x, tx, prep.cfg)
end

function DI.hvp!(
        f,
        tg::NTuple,
        prep::HyperHessiansHVPPrep,
        backend::DI.AutoHyperHessians,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    return hvp!(tg, fc, x, tx, prep.cfg)
end

function DI.gradient_and_hvp(
        f,
        prep::HyperHessiansHVPPrep,
        backend::DI.AutoHyperHessians,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    res = hvp_gradient_value(fc, x, tx, prep.cfg)
    return res.gradient, res.hvp
end

function DI.gradient_and_hvp!(
        f,
        grad,
        tg::NTuple,
        prep::HyperHessiansHVPPrep,
        backend::DI.AutoHyperHessians,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    contexts_prepared = translate_prepared(contexts, prep.contexts_prepared)
    fc = DI.fix_tail(f, contexts_prepared...)
    hvp_gradient_value!(tg, grad, fc, x, tx, prep.cfg)
    return grad, tg
end

end
