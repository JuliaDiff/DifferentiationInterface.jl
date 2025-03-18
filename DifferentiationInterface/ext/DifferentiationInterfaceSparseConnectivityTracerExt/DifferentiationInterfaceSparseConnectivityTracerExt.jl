module DifferentiationInterfaceSparseConnectivityTracerExt

using ADTypes: jacobian_sparsity, hessian_sparsity
import DifferentiationInterface as DI
using SparseConnectivityTracer:
    TracerSparsityDetector, TracerLocalSparsityDetector, jacobian_buffer, hessian_buffer

@inline _translate(::Type, c::DI.Constant) = DI.unwrap(c)
@inline function _translate(::Type{T}, c::DI.Cache) where {T}
    return DI.recursive_similar(DI.unwrap(c), T)
end

function jacobian_translate(detector, x, contexts::Vararg{DI.Context,C}) where {C}
    T = eltype(jacobian_buffer(x, detector))
    new_contexts = map(contexts) do c
        _translate(T, c)
    end
    return new_contexts
end

function hessian_translate(detector, x, contexts::Vararg{DI.Context,C}) where {C}
    T = eltype(hessian_buffer(x, detector))
    new_contexts = map(contexts) do c
        _translate(T, c)
    end
    return new_contexts
end

function DI.jacobian_sparsity_with_contexts(
    f::F,
    detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_tracer = jacobian_translate(detector, x, contexts...)
    fc = DI.FixTail(f, contexts_tracer...)
    return jacobian_sparsity(fc, x, detector)
end

function DI.jacobian_sparsity_with_contexts(
    f!::F,
    y,
    detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_tracer = jacobian_translate(detector, x, contexts...)
    fc! = DI.FixTail(f!, contexts_tracer...)
    return jacobian_sparsity(fc!, y, x, detector)
end

function DI.hessian_sparsity_with_contexts(
    f::F,
    detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_tracer = hessian_translate(detector, x, contexts...)
    fc = DI.FixTail(f, contexts_tracer...)
    return hessian_sparsity(fc, x, detector)
end

end
