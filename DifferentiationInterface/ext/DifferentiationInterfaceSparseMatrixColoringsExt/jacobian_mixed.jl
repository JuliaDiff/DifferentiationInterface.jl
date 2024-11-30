## Preparation

struct MixedModeSparseJacobianPrep{
    BSf<:BatchSizeSettings,
    BSr<:BatchSizeSettings,
    C<:AbstractColoringResult{:nonsymmetric,:bidirectional},
    M<:AbstractMatrix{<:Number},
    Sf<:Vector{<:NTuple},
    Sr<:Vector{<:NTuple},
    Rf<:Vector{<:NTuple},
    Rr<:Vector{<:NTuple},
    Ef<:PushforwardPrep,
    Er<:PullbackPrep,
} <: SparseJacobianPrep
    batch_size_settings_forward::BSf
    batch_size_settings_reverse::BSr
    coloring_result::C
    compressed_matrix_forward::M
    compressed_matrix_reverse::M
    batched_seeds_forward::Sf
    batched_seeds_reverse::Sr
    batched_results_forward::Rf
    batched_results_reverse::Rr
    pushforward_prep::Ef
    pullback_prep::Er
end

function DI.prepare_jacobian(
    f::F, backend::AutoSparse{<:MixedMode}, x, contexts::Vararg{Context,C}
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    return _prepare_mixed_sparse_jacobian_aux(y, (f,), backend, x, contexts...)
end

function DI.prepare_jacobian(
    f!::F, y, backend::AutoSparse{<:MixedMode}, x, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_mixed_sparse_jacobian_aux(y, (f!, y), backend, x, contexts...)
end

function _prepare_mixed_sparse_jacobian_aux(
    y, f_or_f!y::FY, backend::AutoSparse{<:MixedMode}, x, contexts::Vararg{Context,C}
) where {FY,C}
    dense_backend = dense_ad(backend)
    sparsity = jacobian_sparsity(
        fy_with_contexts(f_or_f!y..., contexts...)..., x, sparsity_detector(backend)
    )
    problem = ColoringProblem{:nonsymmetric,:bidirectional}()
    coloring_result = coloring(
        sparsity,
        problem,
        coloring_algorithm(backend);
        decompression_eltype=promote_type(eltype(x), eltype(y)),
    )

    Nf = length(column_groups(coloring_result))
    Nr = length(row_groups(coloring_result))
    batch_size_settings_forward = pick_batchsize(forward_backend(dense_backend), Nf)
    batch_size_settings_reverse = pick_batchsize(reverse_backend(dense_backend), Nr)

    return _prepare_mixed_sparse_jacobian_aux_aux(
        batch_size_settings_forward,
        batch_size_settings_reverse,
        coloring_result,
        y,
        f_or_f!y,
        backend,
        x,
        contexts...,
    )
end

function _prepare_mixed_sparse_jacobian_aux_aux(
    batch_size_settings_forward::BatchSizeSettings{Bf},
    batch_size_settings_reverse::BatchSizeSettings{Br},
    coloring_result::AbstractColoringResult{:nonsymmetric,:bidirectional},
    y,
    f_or_f!y::FY,
    backend::AutoSparse{<:MixedMode},
    x,
    contexts::Vararg{Context,C},
) where {Bf,Br,FY,C}
    Nf, Af = batch_size_settings_forward.N, batch_size_settings_forward.A
    Nr, Ar = batch_size_settings_reverse.N, batch_size_settings_reverse.A

    dense_backend = dense_ad(backend)

    groups_forward = column_groups(coloring_result)
    groups_reverse = row_groups(coloring_result)

    seeds_forward = [
        multibasis(backend, x, eachindex(x)[group]) for group in groups_forward
    ]
    seeds_reverse = [
        multibasis(backend, y, eachindex(y)[group]) for group in groups_reverse
    ]

    compressed_matrix_forward = stack(_ -> vec(similar(y)), groups_forward; dims=2)
    compressed_matrix_reverse = stack(_ -> vec(similar(x)), groups_reverse; dims=1)

    batched_seeds_forward = [
        ntuple(b -> seeds_forward[1 + ((a - 1) * Bf + (b - 1)) % Nf], Val(Bf)) for a in 1:Af
    ]
    batched_seeds_reverse = [
        ntuple(b -> seeds_reverse[1 + ((a - 1) * Br + (b - 1)) % Nr], Val(Br)) for a in 1:Ar
    ]

    batched_results_forward = [
        ntuple(b -> similar(y), Val(Bf)) for _ in batched_seeds_forward
    ]
    batched_results_reverse = [
        ntuple(b -> similar(x), Val(Br)) for _ in batched_seeds_reverse
    ]

    pushforward_prep = prepare_pushforward(
        f_or_f!y...,
        forward_backend(dense_backend),
        x,
        batched_seeds_forward[1],
        contexts...,
    )
    pullback_prep = prepare_pullback(
        f_or_f!y...,
        reverse_backend(dense_backend),
        x,
        batched_seeds_reverse[1],
        contexts...,
    )

    return MixedModeSparseJacobianPrep(
        batch_size_settings_forward,
        batch_size_settings_reverse,
        coloring_result,
        compressed_matrix_forward,
        compressed_matrix_reverse,
        batched_seeds_forward,
        batched_seeds_reverse,
        batched_results_forward,
        batched_results_reverse,
        pushforward_prep,
        pullback_prep,
    )
end

## Common auxiliaries

function _sparse_jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::MixedModeSparseJacobianPrep{<:BatchSizeSettings{Bf},<:BatchSizeSettings{Br}},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,Bf,Br,C}
    (;
        batch_size_settings_forward,
        batch_size_settings_reverse,
        coloring_result,
        compressed_matrix_forward,
        compressed_matrix_reverse,
        batched_seeds_forward,
        batched_seeds_reverse,
        batched_results_forward,
        batched_results_reverse,
        pushforward_prep,
        pullback_prep,
    ) = prep

    dense_backend = dense_ad(backend)
    Nf = batch_size_settings_forward.N
    Nr = batch_size_settings_reverse.N

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y...,
        pushforward_prep,
        forward_backend(dense_backend),
        x,
        batched_seeds_forward[1],
        contexts...,
    )
    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y...,
        pullback_prep,
        reverse_backend(dense_backend),
        x,
        batched_seeds_reverse[1],
        contexts...,
    )

    for a in eachindex(batched_seeds_forward, batched_results_forward)
        pushforward!(
            f_or_f!y...,
            batched_results_forward[a],
            pushforward_prep_same,
            forward_backend(dense_backend),
            x,
            batched_seeds_forward[a],
            contexts...,
        )

        for b in eachindex(batched_results_forward[a])
            copyto!(
                view(compressed_matrix_forward, :, 1 + ((a - 1) * Bf + (b - 1)) % Nf),
                vec(batched_results_forward[a][b]),
            )
        end
    end

    for a in eachindex(batched_seeds_reverse, batched_results_reverse)
        pullback!(
            f_or_f!y...,
            batched_results_reverse[a],
            pullback_prep_same,
            reverse_backend(dense_backend),
            x,
            batched_seeds_reverse[a],
            contexts...,
        )

        for b in eachindex(batched_results_reverse[a])
            copyto!(
                view(compressed_matrix_reverse, 1 + ((a - 1) * Br + (b - 1)) % Nr, :),
                vec(batched_results_reverse[a][b]),
            )
        end
    end

    decompress!(jac, compressed_matrix_reverse, compressed_matrix_forward, coloring_result)

    return jac
end
