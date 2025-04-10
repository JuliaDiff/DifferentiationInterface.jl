struct SMCSparseHessianPrep{
    SIG,
    BS<:DI.BatchSizeSettings,
    P<:AbstractMatrix,
    C<:AbstractColoringResult{:symmetric,:column},
    M<:AbstractMatrix{<:Number},
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    E2<:DI.HVPPrep,
    E1<:DI.GradientPrep,
} <: DI.SparseHessianPrep{SIG}
    _sig::Val{SIG}
    batch_size_settings::BS
    sparsity::P
    coloring_result::C
    compressed_matrix::M
    batched_seeds::S
    batched_results::R
    hvp_prep::E2
    gradient_prep::E1
end

## Hessian, one argument

function DI.prepare_hessian_nokwarg(
    strict::Val, f::F, backend::AutoSparse, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    dense_backend = dense_ad(backend)
    sparsity = DI.hessian_sparsity_with_contexts(
        f, sparsity_detector(backend), x, contexts...
    )
    problem = ColoringProblem{:symmetric,:column}()
    coloring_result = coloring(
        sparsity, problem, coloring_algorithm(backend); decompression_eltype=eltype(x)
    )
    N = length(column_groups(coloring_result))
    batch_size_settings = DI.pick_batchsize(DI.outer(dense_backend), N)
    return _prepare_sparse_hessian_aux(
        strict, batch_size_settings, sparsity, coloring_result, f, backend, x, contexts...
    )
end

function _prepare_sparse_hessian_aux(
    strict::Val,
    batch_size_settings::DI.BatchSizeSettings{B},
    sparsity::AbstractMatrix,
    coloring_result::AbstractColoringResult{:symmetric,:column},
    f::F,
    backend::AutoSparse,
    x,
    contexts::Vararg{DI.Context,C};
) where {B,F,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    (; N, A) = batch_size_settings
    dense_backend = dense_ad(backend)
    groups = column_groups(coloring_result)
    seeds = [DI.multibasis(x, eachindex(x)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(x)), groups; dims=2)
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    hvp_prep = DI.prepare_hvp_nokwarg(
        strict, f, dense_backend, x, batched_seeds[1], contexts...
    )
    gradient_prep = DI.prepare_gradient_nokwarg(
        strict, f, DI.inner(dense_backend), x, contexts...
    )
    return SMCSparseHessianPrep(
        _sig,
        batch_size_settings,
        sparsity,
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        hvp_prep,
        gradient_prep,
    )
end

function DI.hessian!(
    f::F,
    hess,
    prep::SMCSparseHessianPrep{SIG,<:DI.BatchSizeSettings{B}},
    backend::AutoSparse,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (;
        batch_size_settings,
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        hvp_prep,
    ) = prep
    (; N) = batch_size_settings
    dense_backend = dense_ad(backend)

    hvp_prep_same = DI.prepare_hvp_same_point(
        f, hvp_prep, dense_backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        DI.hvp!(
            f,
            batched_results[a],
            hvp_prep_same,
            dense_backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % N),
                vec(batched_results[a][b]),
            )
        end
    end

    decompress!(hess, compressed_matrix, coloring_result)
    return hess
end

function DI.hessian(
    f::F, prep::SMCSparseHessianPrep, backend::AutoSparse, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    hess = similar(sparsity_pattern(prep), eltype(x))
    return DI.hessian!(f, hess, prep, backend, x, contexts...)
end

function DI.value_gradient_and_hessian!(
    f::F,
    grad,
    hess,
    prep::SMCSparseHessianPrep,
    backend::AutoSparse,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, _ = DI.value_and_gradient!(
        f, grad, prep.gradient_prep, DI.inner(dense_ad(backend)), x, contexts...
    )
    DI.hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian(
    f::F, prep::SMCSparseHessianPrep, backend::AutoSparse, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, grad = DI.value_and_gradient(
        f, prep.gradient_prep, DI.inner(dense_ad(backend)), x, contexts...
    )
    hess = DI.hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end
