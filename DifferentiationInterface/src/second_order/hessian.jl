## Docstrings

"""
    prepare_hessian(f, backend, x, [contexts...]; strict=Val(true)) -> prep

$(docstring_prepare("hessian"))
"""
function prepare_hessian(
        f::F, backend::AbstractADType, x, contexts::Vararg{Context, C}; strict::Val = Val(true)
    ) where {F, C}
    return prepare_hessian_nokwarg(strict, f, backend, x, contexts...)
end

"""
    prepare!_hessian(f, backend, x, [contexts...]) -> new_prep

$(docstring_prepare!("hessian"))
"""
function prepare!_hessian(
        f::F, old_prep::HessianPrep, backend::AbstractADType, x, contexts::Vararg{Context, C}
    ) where {F, C}
    check_prep(f, old_prep, backend, x, contexts...)
    return prepare_hessian_nokwarg(is_strict(old_prep), f, backend, x, contexts...)
end

"""
    hessian(f, [prep,] backend, x, [contexts...]) -> hess

Compute the Hessian matrix of the function `f` at point `x`.

$(docstring_preparation_hint("hessian"))
"""
function hessian(f::F, backend::AbstractADType, x, contexts::Vararg{Context, C}) where {F, C}
    prep = prepare_hessian_nokwarg(Val(true), f, backend, x, contexts...)
    return hessian(f, prep, backend, x, contexts...)
end

"""
    hessian!(f, hess, [prep,] backend, x, [contexts...]) -> hess

Compute the Hessian matrix of the function `f` at point `x`, overwriting `hess`.

$(docstring_preparation_hint("hessian"))
"""
function hessian!(
        f::F, hess, backend::AbstractADType, x, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hessian_nokwarg(Val(true), f, backend, x, contexts...)
    return hessian!(f, hess, prep, backend, x, contexts...)
end

"""
    value_gradient_and_hessian(f, [prep,] backend, x, [contexts...]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`.

$(docstring_preparation_hint("hessian"))
"""
function value_gradient_and_hessian(
        f::F, backend::AbstractADType, x, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hessian_nokwarg(Val(true), f, backend, x, contexts...)
    return value_gradient_and_hessian(f, prep, backend, x, contexts...)
end

"""
    value_gradient_and_hessian!(f, grad, hess, [prep,] backend, x, [contexts...]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.

$(docstring_preparation_hint("hessian"))
"""
function value_gradient_and_hessian!(
        f::F, grad, hess, backend::AbstractADType, x, contexts::Vararg{Context, C}
    ) where {F, C}
    prep = prepare_hessian_nokwarg(Val(true), f, backend, x, contexts...)
    return value_gradient_and_hessian!(f, grad, hess, prep, backend, x, contexts...)
end

## Preparation

struct HVPGradientHessianPrep{
        SIG,
        BS <: BatchSizeSettings,
        S <: AbstractVector{<:NTuple},
        R <: AbstractVector{<:NTuple},
        SE <: NTuple,
        E2 <: HVPPrep,
        E1 <: GradientPrep,
    } <: HessianPrep{SIG}
    _sig::Val{SIG}
    batch_size_settings::BS
    batched_seeds::S
    batched_results::R
    seed_example::SE
    hvp_prep::E2
    gradient_prep::E1
end

function prepare_hessian_nokwarg(
        strict::Val, f::F, backend::AbstractADType, x, contexts::Vararg{Context, C}
    ) where {F, C}
    # type-unstable
    batch_size_settings = pick_batchsize(outer(backend), x)
    # function barrier
    return _prepare_hessian_aux(strict, batch_size_settings, f, backend, x, contexts...)
end

function _prepare_hessian_aux(
        strict::Val,
        batch_size_settings::BatchSizeSettings{B},
        f::F,
        backend::AbstractADType,
        x,
        contexts::Vararg{Context, C}
    ) where {B, F, C}
    _sig = signature(f, backend, x, contexts...; strict)
    (; N, A) = batch_size_settings
    seeds = [basis(x, ind) for ind in eachindex(x)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    seed_example = ntuple(b -> basis(x), Val(B))
    hvp_prep = prepare_hvp_nokwarg(strict, f, backend, x, seed_example, contexts...)
    gradient_prep = prepare_gradient_nokwarg(strict, f, inner(backend), x, contexts...)
    return HVPGradientHessianPrep(
        _sig,
        batch_size_settings,
        batched_seeds,
        batched_results,
        seed_example,
        hvp_prep,
        gradient_prep,
    )
end

## One argument

function hessian(
        f::F,
        prep::HVPGradientHessianPrep{SIG, <:BatchSizeSettings{B, true}},
        backend::AbstractADType,
        x,
        contexts::Vararg{Context, C},
    ) where {F, SIG, B, C}
    check_prep(f, prep, backend, x, contexts...)
    (; batched_seeds, hvp_prep) = prep
    dg_batch = hvp(f, hvp_prep, backend, x, only(batched_seeds), contexts...)
    block = stack_vec_col(dg_batch)
    return block
end

function hessian(
        f::F,
        prep::HVPGradientHessianPrep{SIG, <:BatchSizeSettings{1, false, true}},
        backend::AbstractADType,
        x,
        contexts::Vararg{Context, C},
    ) where {F, SIG, C}
    check_prep(f, prep, backend, x, contexts...)
    (; batched_seeds, seed_example, hvp_prep) = prep

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, seed_example, contexts...
    )

    hess = mapreduce(hcat, eachindex(batched_seeds)) do a
        dg = only(hvp(f, hvp_prep_same, backend, x, batched_seeds[a], contexts...))
        return vec(dg)
    end
    return hess
end

function hessian(
        f::F,
        prep::HVPGradientHessianPrep{SIG, <:BatchSizeSettings{B, false, aligned}},
        backend::AbstractADType,
        x,
        contexts::Vararg{Context, C},
    ) where {F, SIG, B, aligned, C}
    check_prep(f, prep, backend, x, contexts...)
    (; batch_size_settings, batched_seeds, seed_example, hvp_prep) = prep
    (; A, B_last) = batch_size_settings

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, seed_example, contexts...
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
        prep::HVPGradientHessianPrep{SIG, <:BatchSizeSettings{B}},
        backend::AbstractADType,
        x,
        contexts::Vararg{Context, C},
    ) where {F, SIG, B, C}
    check_prep(f, prep, backend, x, contexts...)
    (; batch_size_settings, batched_seeds, batched_results, seed_example, hvp_prep) = prep
    (; N) = batch_size_settings

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, seed_example, contexts...
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
        contexts::Vararg{Context, C},
    ) where {F, C}
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
        contexts::Vararg{Context, C},
    ) where {F, C}
    check_prep(f, prep, backend, x, contexts...)
    y, _ = value_and_gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end
