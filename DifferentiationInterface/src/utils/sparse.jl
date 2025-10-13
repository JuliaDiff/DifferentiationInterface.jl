"""
    jacobian_sparsity_with_contexts(f, detector, x, contexts...)
    jacobian_sparsity_with_contexts(f!, y, detector, x, contexts...)

Wrapper around [`ADTypes.jacobian_sparsity`](@extref ADTypes.jacobian_sparsity) enabling the allocation of caches with proper element types.
"""
function jacobian_sparsity_with_contexts(
        f::F, detector::AbstractSparsityDetector, x, contexts::Vararg{Context, C}
    ) where {F, C}
    return jacobian_sparsity(fix_tail(f, map(unwrap, contexts)...), x, detector)
end

function jacobian_sparsity_with_contexts(
        f!::F, y, detector::AbstractSparsityDetector, x, contexts::Vararg{Context, C}
    ) where {F, C}
    return jacobian_sparsity(fix_tail(f!, map(unwrap, contexts)...), y, x, detector)
end

"""
    hessian_sparsity_with_contexts(f, detector, x, contexts...)

Wrapper around [`ADTypes.hessian_sparsity`](@extref ADTypes.hessian_sparsity) enabling the allocation of caches with proper element types.
"""
function hessian_sparsity_with_contexts(
        f::F, detector::AbstractSparsityDetector, x, contexts::Vararg{Context, C}
    ) where {F, C}
    return hessian_sparsity(fix_tail(f, map(unwrap, contexts)...), x, detector)
end
