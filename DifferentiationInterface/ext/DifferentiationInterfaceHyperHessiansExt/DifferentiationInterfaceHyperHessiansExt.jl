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
DI.hvp_mode(::DI.AutoHyperHessians) = DI.ForwardOverForward()
DI.mode(::DI.AutoHyperHessians) = ForwardMode()

chunk_from_backend(backend::DI.AutoHyperHessians, x) =
    isnothing(backend.chunksize) ? Chunk(x) : Chunk{backend.chunksize}()
chunk_from_backend(backend::DI.AutoHyperHessians, N::Integer, ::Type{T}) where {T} =
    isnothing(backend.chunksize) ? Chunk(pickchunksize(N, T), T) : Chunk{backend.chunksize}()

function DI.pick_batchsize(backend::DI.AutoHyperHessians, x::AbstractArray)
    B = chunksize(chunk_from_backend(backend, x))
    return DI.BatchSizeSettings{B}(length(x))
end

## Second derivative (scalar input)

struct HyperHessiansSecondDerivativePrep{SIG} <: DI.SecondDerivativePrep{SIG}
    _sig::Val{SIG}
end

function DI.prepare_second_derivative_nokwarg(
        strict::Val, f, backend::DI.AutoHyperHessians, x::Number, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return HyperHessiansSecondDerivativePrep(_sig)
end

function DI.second_derivative(
        f,
        prep::HyperHessiansSecondDerivativePrep,
        backend::DI.AutoHyperHessians,
        x::Number,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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

struct HyperHessiansHessianPrep{SIG, C} <: DI.HessianPrep{SIG}
    _sig::Val{SIG}
    cfg::C
end

struct HyperHessiansHVPPrep{SIG, C} <: DI.HVPPrep{SIG}
    _sig::Val{SIG}
    cfg::C
end

## Hessian

function DI.prepare_hessian_nokwarg(
        strict::Val, f, backend::DI.AutoHyperHessians, x::AbstractArray, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    cfg = HessianConfig(x, chunk_from_backend(backend, x))
    return HyperHessiansHessianPrep(_sig, cfg)
end

function DI.hessian(
        f,
        prep::HyperHessiansHessianPrep,
        backend::DI.AutoHyperHessians,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    val = hessian_gradient_value!(hess, grad, fc, x, prep.cfg)
    return val, grad, hess
end

## HVP

function DI.prepare_hvp_nokwarg(
        strict::Val, f, backend::DI.AutoHyperHessians, x::AbstractArray, tx::NTuple, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    cfg = HVPConfig(x, tx, chunk_from_backend(backend, x))
    return HyperHessiansHVPPrep(_sig, cfg)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    hvp_gradient_value!(tg, grad, fc, x, tx, prep.cfg)
    return grad, tg
end

end
