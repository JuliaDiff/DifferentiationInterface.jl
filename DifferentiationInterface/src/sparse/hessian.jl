struct SparseHessianExtras{
    B,
    C<:AbstractColoringResult{:symmetric,:column},
    M<:AbstractMatrix{<:Real},
    D,
    R,
    E2<:HVPExtras,
    E1<:GradientExtras,
} <: HessianExtras
    coloring_result::C
    compressed_matrix::M
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    hvp_extras::E2
    gradient_extras::E1
end

function SparseHessianExtras{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Tangents{B,D}},
    batched_results::Vector{Tangents{B,R}},
    hvp_extras::E2,
    gradient_extras::E1,
) where {B,C,M,D,R,E2,E1}
    return SparseHessianExtras{B,C,M,D,R,E2,E1}(
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        hvp_extras,
        gradient_extras,
    )
end

## Hessian, one argument

function prepare_hessian(f::F, backend::AutoSparse, x) where {F}
    dense_backend = dense_ad(backend)
    sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    problem = ColoringProblem{:symmetric,:column}()
    coloring_result = coloring(
        sparsity, problem, coloring_algorithm(backend); decompression_eltype=eltype(x)
    )
    groups = column_groups(coloring_result)
    Ng = length(groups)
    B = pick_batchsize(maybe_outer(dense_backend), Ng)
    seeds = [multibasis(backend, x, CartesianIndices(x)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(x)), groups; dims=2)
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B))) for
        a in 1:div(Ng, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))) for _ in batched_seeds]
    hvp_extras = prepare_hvp(f, dense_backend, x, batched_seeds[1])
    gradient_extras = prepare_gradient(f, maybe_inner(dense_backend), x)
    return SparseHessianExtras{B}(;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        hvp_extras,
        gradient_extras,
    )
end

function hessian(f::F, backend::AutoSparse, x, extras::SparseHessianExtras{B}) where {F,B}
    @compat (; coloring_result, batched_seeds, hvp_extras) = extras
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    hvp_extras_same = prepare_hvp_same_point(
        f, dense_backend, x, batched_seeds[1], hvp_extras
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dg_batch = hvp(f, dense_backend, x, batched_seeds[a], hvp_extras_same)
        stack(vec, dg_batch.d; dims=2)
    end

    compressed_matrix = reduce(hcat, compressed_blocks)
    if Ng < size(compressed_matrix, 2)
        compressed_matrix = compressed_matrix[:, 1:Ng]
    end
    return decompress(compressed_matrix, coloring_result)
end

function hessian!(
    f::F, hess, backend::AutoSparse, x, extras::SparseHessianExtras{B}
) where {F,B}
    @compat (;
        coloring_result, compressed_matrix, batched_seeds, batched_results, hvp_extras
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    hvp_extras_same = prepare_hvp_same_point(
        f, dense_backend, x, batched_seeds[1], hvp_extras
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp!(f, batched_results[a], dense_backend, x, batched_seeds[a], hvp_extras_same)

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % Ng),
                vec(batched_results[a].d[b]),
            )
        end
    end

    decompress!(hess, compressed_matrix, coloring_result)
    return hess
end

function value_gradient_and_hessian!(
    f::F, grad, hess, backend::AutoSparse, x, extras::SparseHessianExtras
) where {F}
    y, _ = value_and_gradient!(
        f, grad, maybe_inner(dense_ad(backend)), x, extras.gradient_extras
    )
    hessian!(f, hess, backend, x, extras)
    return y, grad, hess
end

function value_gradient_and_hessian(
    f::F, backend::AutoSparse, x, extras::SparseHessianExtras
) where {F}
    y, grad = value_and_gradient(
        f, maybe_inner(dense_ad(backend)), x, extras.gradient_extras
    )
    hess = hessian(f, backend, x, extras)
    return y, grad, hess
end
