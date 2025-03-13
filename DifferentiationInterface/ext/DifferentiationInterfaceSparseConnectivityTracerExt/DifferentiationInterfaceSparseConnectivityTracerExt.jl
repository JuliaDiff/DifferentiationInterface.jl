module DifferentiationInterfaceSparseConnectivityTracerExt

using ADTypes: jacobian_sparsity, hessian_sparsity
import DifferentiationInterface as DI
using SparseConnectivityTracer:
    TracerSparsityDetector, TracerLocalSparsityDetector, jacobian_buffer, hessian_buffer

@inline _jacobian_translate(detector, c::DI.Constant) = DI.unwrap(c)
@inline function _jacobian_translate(detector, c::DI.Cache{<:AbstractArray})
    return jacobian_buffer(DI.unwrap(c), detector)
end

function jacobian_translate(detector, contexts::Vararg{DI.Context,C}) where {C}
    new_contexts = map(contexts) do c
        _jacobian_translate(detector, c)
    end
    return new_contexts
end

@inline _hessian_translate(detector, c::DI.Constant) = DI.unwrap(c)
@inline function _hessian_translate(detector, c::DI.Cache{<:AbstractArray})
    return hessian_buffer(DI.unwrap(c), detector)
end

function hessian_translate(detector, contexts::Vararg{DI.Context,C}) where {C}
    new_contexts = map(contexts) do c
        _hessian_translate(detector, c)
    end
    return new_contexts
end

function DI.jacobian_sparsity_with_contexts(
    f::F,
    detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_tracer = jacobian_translate(detector, contexts...)
    fc = DI.FixTail(f, contexts_tracer)
    return jacobian_sparsity(fc, x, detector)
end

function DI.jacobian_sparsity_with_contexts(
    f!::F,
    y,
    detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_tracer = jacobian_translate(detector, contexts...)
    fc! = DI.FixTail(f!, contexts_tracer)
    return jacobian_sparsity(fc!, y, x, detector)
end

function DI.hessian_sparsity_with_contexts(
    f::F,
    detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_tracer = hessian_translate(detector, contexts...)
    fc = DI.FixTail(f, contexts_tracer)
    return hessian_sparsity(fc, x, detector)
end

end
