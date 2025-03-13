function jacobian_sparsity_with_contexts(
    f::F, detector::AbstractSparsityDetector, x, contexts::Vararg{Context,C}
) where {F,C}
    return jacobian_sparsity(with_contexts(f, contexts...), x, detector)
end

function jacobian_sparsity_with_contexts(
    f!::F, y, detector::AbstractSparsityDetector, x, contexts::Vararg{Context,C}
) where {F,C}
    return jacobian_sparsity(with_contexts(f!, contexts...), y, x, detector)
end

function hessian_sparsity_with_contexts(
    f::F, detector::AbstractSparsityDetector, x, contexts::Vararg{Context,C}
) where {F,C}
    return hessian_sparsity(with_contexts(f, contexts...), x, detector)
end
