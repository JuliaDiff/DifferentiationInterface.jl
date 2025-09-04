## Docstrings

"""
    prepare_jacobian(f,     backend, x, [contexts...]; strict=Val(true)) -> prep
    prepare_jacobian(f!, y, backend, x, [contexts...]; strict=Val(true)) -> prep

$(docstring_prepare("jacobian"; inplace=true))
"""
function prepare_jacobian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Val=Val(true)
) where {F,C}
    return prepare_jacobian_nokwarg(strict, f, backend, x, contexts...)
end

function prepare_jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Val=Val(true)
) where {F,C}
    return prepare_jacobian_nokwarg(strict, f!, y, backend, x, contexts...)
end

"""
    prepare!_jacobian(f,     prep, backend, x, [contexts...]) -> new_prep
    prepare!_jacobian(f!, y, prep, backend, x, [contexts...]) -> new_prep

$(docstring_prepare!("jacobian"))
"""
function prepare!_jacobian(
    f::F, old_prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C};
) where {F,C}
    check_prep(f, old_prep, backend, x, contexts...)
    return prepare_jacobian_nokwarg(is_strict(old_prep), f, backend, x, contexts...)
end

function prepare!_jacobian(
    f!::F,
    y,
    old_prep::JacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C};
) where {F,C}
    check_prep(f!, y, old_prep, backend, x, contexts...)
    return prepare_jacobian_nokwarg(is_strict(old_prep), f!, y, backend, x, contexts...)
end

"""
    value_and_jacobian(f,     [prep,] backend, x, [contexts...]) -> (y, jac)
    value_and_jacobian(f!, y, [prep,] backend, x, [contexts...]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`.

$(docstring_preparation_hint("jacobian"))
"""
function value_and_jacobian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f, backend, x, contexts...)
    return value_and_jacobian(f, prep, backend, x, contexts...)
end

function value_and_jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f!, y, backend, x, contexts...)
    return value_and_jacobian(f!, y, prep, backend, x, contexts...)
end

"""
    value_and_jacobian!(f,     jac, [prep,] backend, x, [contexts...]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, [prep,] backend, x, [contexts...]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.

$(docstring_preparation_hint("jacobian"))
"""
function value_and_jacobian!(
    f::F, jac, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f, backend, x, contexts...)
    return value_and_jacobian!(f, jac, prep, backend, x, contexts...)
end

function value_and_jacobian!(
    f!::F, y, jac, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f!, y, backend, x, contexts...)
    return value_and_jacobian!(f!, y, jac, prep, backend, x, contexts...)
end

"""
    jacobian(f,     [prep,] backend, x, [contexts...]) -> jac
    jacobian(f!, y, [prep,] backend, x, [contexts...]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`.

$(docstring_preparation_hint("jacobian"))
"""
function jacobian(f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f, backend, x, contexts...)
    return jacobian(f, prep, backend, x, contexts...)
end

function jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f!, y, backend, x, contexts...)
    return jacobian(f!, y, prep, backend, x, contexts...)
end

"""
    jacobian!(f,     jac, [prep,] backend, x, [contexts...]) -> jac
    jacobian!(f!, y, jac, [prep,] backend, x, [contexts...]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.

$(docstring_preparation_hint("jacobian"))
"""
function jacobian!(
    f::F, jac, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f, backend, x, contexts...)
    return jacobian!(f, jac, prep, backend, x, contexts...)
end

function jacobian!(
    f!::F, y, jac, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    prep = prepare_jacobian_nokwarg(Val(true), f!, y, backend, x, contexts...)
    return jacobian!(f!, y, jac, prep, backend, x, contexts...)
end

## Preparation

abstract type StandardJacobianPrep{SIG} <: JacobianPrep{SIG} end

struct PushforwardJacobianPrep{
    SIG,
    BS<:BatchSizeSettings,
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    SE<:NTuple,
    E<:PushforwardPrep,
} <: StandardJacobianPrep{SIG}
    _sig::Val{SIG}
    batch_size_settings::BS
    batched_seeds::S
    batched_results::R
    seed_example::SE
    pushforward_prep::E
end

struct PullbackJacobianPrep{
    SIG,
    BS<:BatchSizeSettings,
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    SE<:NTuple,
    E<:PullbackPrep,
} <: StandardJacobianPrep{SIG}
    _sig::Val{SIG}
    batch_size_settings::BS
    batched_seeds::S
    batched_results::R
    seed_example::SE
    pullback_prep::E
end

function prepare_jacobian_nokwarg(
    strict::Val, f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    perf = pushforward_performance(backend)
    # type-unstable
    if perf isa PushforwardFast
        batch_size_settings = pick_batchsize(backend, x)
    else
        batch_size_settings = pick_batchsize(backend, y)
    end
    # function barrier
    return _prepare_jacobian_aux(
        strict, perf, batch_size_settings, y, (f,), backend, x, contexts...
    )
end

function prepare_jacobian_nokwarg(
    strict::Val, f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C};
) where {F,C}
    perf = pushforward_performance(backend)
    # type-unstable
    if perf isa PushforwardFast
        batch_size_settings = pick_batchsize(backend, x)
    else
        batch_size_settings = pick_batchsize(backend, y)
    end
    # function barrier
    return _prepare_jacobian_aux(
        strict, perf, batch_size_settings, y, (f!, y), backend, x, contexts...
    )
end

function _prepare_jacobian_aux(
    strict::Val,
    ::PushforwardFast,
    batch_size_settings::BatchSizeSettings{B},
    y,
    f_or_f!y::FY,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C};
) where {B,FY,C}
    _sig = signature(f_or_f!y..., backend, x, contexts...; strict)
    (; N, A) = batch_size_settings
    seeds = [basis(x, ind) for ind in eachindex(x)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(y), Val(B)) for _ in batched_seeds]
    seed_example = ntuple(b -> basis(x), Val(B))
    pushforward_prep = prepare_pushforward_nokwarg(
        strict, f_or_f!y..., backend, x, seed_example, contexts...
    )
    return PushforwardJacobianPrep(
        _sig,
        batch_size_settings,
        batched_seeds,
        batched_results,
        seed_example,
        pushforward_prep,
    )
end

function _prepare_jacobian_aux(
    strict::Val,
    ::PushforwardSlow,
    batch_size_settings::BatchSizeSettings{B},
    y,
    f_or_f!y::FY,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C};
) where {B,FY,C}
    _sig = signature(f_or_f!y..., backend, x, contexts...; strict)
    (; N, A) = batch_size_settings
    seeds = [basis(y, ind) for ind in eachindex(y)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    seed_example = ntuple(b -> basis(y), Val(B))
    pullback_prep = prepare_pullback_nokwarg(
        strict, f_or_f!y..., backend, x, seed_example, contexts...
    )
    return PullbackJacobianPrep(
        _sig,
        batch_size_settings,
        batched_seeds,
        batched_results,
        seed_example,
        pullback_prep,
    )
end

## One argument

function jacobian(
    f::F,
    prep::StandardJacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
    return _jacobian_aux((f,), prep, backend, x, contexts...)
end

function jacobian!(
    f::F,
    jac,
    prep::StandardJacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
    return _jacobian_aux!((f,), jac, prep, backend, x, contexts...)
end

function value_and_jacobian(
    f::F, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
    return f(x, map(unwrap, contexts)...), jacobian(f, prep, backend, x, contexts...)
end

function value_and_jacobian!(
    f::F, jac, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    check_prep(f, prep, backend, x, contexts...)
    return f(x, map(unwrap, contexts)...), jacobian!(f, jac, prep, backend, x, contexts...)
end

## Two arguments

function jacobian(
    f!::F,
    y,
    prep::StandardJacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, contexts...)
    return _jacobian_aux((f!, y), prep, backend, x, contexts...)
end

function jacobian!(
    f!::F,
    y,
    jac,
    prep::StandardJacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, contexts...)
    return _jacobian_aux!((f!, y), jac, prep, backend, x, contexts...)
end

function value_and_jacobian(
    f!::F, y, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    check_prep(f!, y, prep, backend, x, contexts...)
    jac = jacobian(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

function value_and_jacobian!(
    f!::F,
    y,
    jac,
    prep::JacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    check_prep(f!, y, prep, backend, x, contexts...)
    jacobian!(f!, y, jac, prep, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

## Common auxiliaries

function _jacobian_aux(
    f_or_f!y::FY,
    prep::PushforwardJacobianPrep{SIG,<:BatchSizeSettings{B,true,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,SIG,B,aligned,C}
    (; batch_size_settings, batched_seeds, pushforward_prep) = prep
    (; B_last) = batch_size_settings
    dy_batch = pushforward(
        f_or_f!y..., pushforward_prep, backend, x, only(batched_seeds), contexts...
    )
    block = stack_vec_col(dy_batch)
    if aligned
        return block
    else
        return block[:, 1:B_last]
    end
end

function _jacobian_aux(
    f_or_f!y::FY,
    prep::PushforwardJacobianPrep{SIG,<:BatchSizeSettings{B,false,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,SIG,B,aligned,C}
    (; batch_size_settings, batched_seeds, seed_example, pushforward_prep) = prep
    (; A, B_last) = batch_size_settings

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, backend, x, seed_example, contexts...
    )

    jac = mapreduce(hcat, eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y...,
            pushforward_prep_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )
        block = stack_vec_col(dy_batch)
        if !aligned && a == A
            return block[:, 1:B_last]
        else
            return block
        end
    end
    return jac
end

function _jacobian_aux(
    f_or_f!y::FY,
    prep::PullbackJacobianPrep{SIG,<:BatchSizeSettings{B,true,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,SIG,B,aligned,C}
    (; batch_size_settings, batched_seeds, pullback_prep) = prep
    (; B_last) = batch_size_settings
    dx_batch = pullback(
        f_or_f!y..., pullback_prep, backend, x, only(batched_seeds), contexts...
    )
    if eltype(x) <: Complex
        dx_batch = map(conj, dx_batch)
    end
    block = stack_vec_row(dx_batch)
    if aligned
        return block
    else
        return block[1:B_last, :]
    end
end

function _jacobian_aux(
    f_or_f!y::FY,
    prep::PullbackJacobianPrep{SIG,<:BatchSizeSettings{B,false,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,SIG,B,aligned,C}
    (; batch_size_settings, batched_seeds, seed_example, pullback_prep) = prep
    (; A, B_last) = batch_size_settings

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_prep, backend, x, seed_example, contexts...
    )

    jac = mapreduce(vcat, eachindex(batched_seeds)) do a
        dx_batch = pullback(
            f_or_f!y..., pullback_prep_same, backend, x, batched_seeds[a], contexts...
        )
        if eltype(x) <: Complex
            dx_batch = map(conj, dx_batch)
        end
        block = stack_vec_row(dx_batch)
        if !aligned && a == A
            return block[1:B_last, :]
        else
            return block
        end
    end
    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PushforwardJacobianPrep{SIG,<:BatchSizeSettings{B}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,SIG,B,C}
    (;
        batch_size_settings, batched_seeds, batched_results, seed_example, pushforward_prep
    ) = prep
    (; N) = batch_size_settings

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, backend, x, seed_example, contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            pushforward_prep_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a][b])
            )
        end
    end

    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PullbackJacobianPrep{SIG,<:BatchSizeSettings{B}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,SIG,B,C}
    (; batch_size_settings, batched_seeds, batched_results, seed_example, pullback_prep) =
        prep
    (; N) = batch_size_settings

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_prep, backend, x, seed_example, contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            pullback_prep_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            if eltype(x) <: Complex
                batched_results[a][b] .= conj.(batched_results[a][b])
            end
            copyto!(
                view(jac, 1 + ((a - 1) * B + (b - 1)) % N, :), vec(batched_results[a][b])
            )
        end
    end

    return jac
end
