## Docstrings

"""
    prepare_hessian(f, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`hessian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hessian end

"""
    prepare!_hessian(f, backend, x, [contexts...]) -> new_prep

Same behavior as [`prepare_hessian`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_hessian end

"""
    hessian(f, [prep,] backend, x, [contexts...]) -> hess

Compute the Hessian matrix of the function `f` at point `x`.

$(document_preparation("hessian"))
"""
function hessian end

"""
    hessian!(f, hess, [prep,] backend, x, [contexts...]) -> hess

Compute the Hessian matrix of the function `f` at point `x`, overwriting `hess`.

$(document_preparation("hessian"))
"""
function hessian! end

"""
    value_gradient_and_hessian(f, [prep,] backend, x, [contexts...]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian end

"""
    value_gradient_and_hessian!(f, grad, hess, [prep,] backend, x, [contexts...]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian! end

## Preparation

struct HVPGradientHessianPrep{
    BS<:BatchSizeSettings,
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    E2<:HVPPrep,
    E1<:GradientPrep,
} <: HessianPrep
    batch_size_settings::BS
    batched_seeds::S
    batched_results::R
    hvp_prep::E2
    gradient_prep::E1
end

function prepare_hessian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    # type-unstable
    batch_size_settings = pick_batchsize(outer(backend), x)
    # function barrier
    return _prepare_hessian_aux(batch_size_settings, f, backend, x, contexts...)
end

function _prepare_hessian_aux(
    batch_size_settings::BatchSizeSettings{B},
    f::F,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {B,F,C}
    (; N, A) = batch_size_settings
    seeds = [basis(x, ind) for ind in eachindex(x)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    hvp_prep = prepare_hvp(f, backend, x, batched_seeds[1], contexts...)
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    return HVPGradientHessianPrep(
        batch_size_settings, batched_seeds, batched_results, hvp_prep, gradient_prep
    )
end

## One argument

function hessian(
    f::F,
    prep::HVPGradientHessianPrep{<:BatchSizeSettings{B,true}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,B,C}
    check_prep(f, prep, backend, x, contexts...)
    (; batched_seeds, hvp_prep) = prep
    dg_batch = hvp(f, hvp_prep, backend, x, only(batched_seeds), contexts...)
    block = stack_vec_col(dg_batch)
    return block
end

function hessian(
    f::F,
    prep::HVPGradientHessianPrep{<:BatchSizeSettings{B,false,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,B,aligned,C}
    check_prep(f, prep, backend, x, contexts...)
    (; batch_size_settings, batched_seeds, hvp_prep) = prep
    (; A, B_last) = batch_size_settings

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, batched_seeds[1], contexts...
    )

    hess = mapreduce(hcat, eachindex(batched_seeds)) do a
        dg_batch = hvp(f, hvp_prep_same, backend, x, batched_seeds[a], contexts...)
        block = stack_vec_col(dg_batch)
        if !aligned && a == A
            return block[:, 1:B_last]
        else
            return block
        end
    end
    return hess
end

function hessian!(
    f::F,
    hess,
    prep::HVPGradientHessianPrep{<:BatchSizeSettings{B}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,B,C}
    check_prep(f, prep, backend, x, contexts...)
    (; batch_size_settings, batched_seeds, batched_results, hvp_prep) = prep
    (; N) = batch_size_settings

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp!(
            f, batched_results[a], hvp_prep_same, backend, x, batched_seeds[a], contexts...
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(hess, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a][b])
            )
        end
    end

    return hess
end

function value_gradient_and_hessian(
    f::F,
    prep::HVPGradientHessianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
    y, grad = value_and_gradient(f, prep.gradient_prep, inner(backend), x, contexts...)
    hess = hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end

function value_gradient_and_hessian!(
    f::F,
    grad,
    hess,
    prep::HVPGradientHessianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
    y, _ = value_and_gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end
