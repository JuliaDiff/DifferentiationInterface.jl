## Docstrings

"""
    prepare_jacobian(f,     backend, x, [contexts...]) -> prep
    prepare_jacobian(f!, y, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`jacobian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_jacobian end

"""
    prepare!_jacobian(f,     prep, backend, x, [contexts...]) -> new_prep
    prepare!_jacobian(f!, y, prep, backend, x, [contexts...]) -> new_prep

Same behavior as [`prepare_jacobian`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_jacobian end

"""
    value_and_jacobian(f,     [prep,] backend, x, [contexts...]) -> (y, jac)
    value_and_jacobian(f!, y, [prep,] backend, x, [contexts...]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function value_and_jacobian end

"""
    value_and_jacobian!(f,     jac, [prep,] backend, x, [contexts...]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, [prep,] backend, x, [contexts...]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.
    
$(document_preparation("jacobian"))
"""
function value_and_jacobian! end

"""
    jacobian(f,     [prep,] backend, x, [contexts...]) -> jac
    jacobian(f!, y, [prep,] backend, x, [contexts...]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function jacobian end

"""
    jacobian!(f,     jac, [prep,] backend, x, [contexts...]) -> jac
    jacobian!(f!, y, jac, [prep,] backend, x, [contexts...]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.

$(document_preparation("jacobian"))
"""
function jacobian! end

## Preparation

abstract type StandardJacobianPrep{F,Y,B,X,CC} <: JacobianPrep{F,Y,B,X,CC} end

struct PushforwardJacobianPrep{
    F,
    Y,
    B,
    X,
    CC,
    BS<:BatchSizeSettings,
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    E<:PushforwardPrep,
} <: StandardJacobianPrep{F,Y,B,X,CC}
    batch_size_settings::BS
    batched_seeds::S
    batched_results::R
    pushforward_prep::E
end

function PushforwardJacobianPrep{F,Y,B,X,CC}(
    batch_size_settings::BS, batched_seeds::S, batched_results::R, pushforward_prep::E
) where {F,Y,B,X,CC}
    return PushforwardJacobianPrep{F,Y,B,X,CC,BS,S,R,E}(
        batch_size_settings, batched_seeds, batched_results, pushforward_prep
    )
end

struct PullbackJacobianPrep{
    F,
    Y,
    B,
    X,
    CC,
    BS<:BatchSizeSettings,
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    E<:PullbackPrep,
} <: StandardJacobianPrep
    batch_size_settings::BS
    batched_seeds::S
    batched_results::R
    pullback_prep::E
end

function PullbackJacobianPrep{F,Y,B,X,CC}(
    batch_size_settings::BS, batched_seeds::S, batched_results::R, pullback_prep::E
) where {F,Y,B,X,CC}
    return PullbackJacobianPrep{F,Y,B,X,CC,BS,S,R,E}(
        batch_size_settings, batched_seeds, batched_results, pullback_prep
    )
end

function prepare_jacobian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
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
        perf, batch_size_settings, y, (f,), backend, x, contexts...
    )
end

function prepare_jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
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
        perf, batch_size_settings, y, (f!, y), backend, x, contexts...
    )
end

function _prepare_jacobian_aux(
    ::PushforwardFast,
    batch_size_settings::BatchSizeSettings{B},
    y,
    f_or_f!y::FY,
    backend::AD,
    x::X,
    contexts::Vararg{Context,C},
) where {B,FY,AD,X,C}
    (; N, A) = batch_size_settings
    seeds = [basis(x, ind) for ind in eachindex(x)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(y), Val(B)) for _ in batched_seeds]
    pushforward_prep = prepare_pushforward(
        f_or_f!y..., backend, x, batched_seeds[1], contexts...
    )
    F, Y = if f_or_f!y isa Tuple
        typeof.(f_or_f!y)
    else
        typeof(f_or_f!y), Nothing
    end
    return PushforwardJacobianPrep{F,Y,AD,X,typeof(contexts)}(
        batch_size_settings, batched_seeds, batched_results, pushforward_prep
    )
end

function _prepare_jacobian_aux(
    ::PushforwardSlow,
    batch_size_settings::BatchSizeSettings{B},
    y,
    f_or_f!y::FY,
    backend::AD,
    x,
    contexts::Vararg{Context,C},
) where {B,FY,AD,C}
    (; N, A) = batch_size_settings
    seeds = [basis(y, ind) for ind in eachindex(y)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    pullback_prep = prepare_pullback(f_or_f!y..., backend, x, batched_seeds[1], contexts...)
    F, Y = if f_or_f!y isa Tuple
        typeof.(f_or_f!y)
    else
        typeof(f_or_f!y), Nothing
    end
    return PullbackJacobianPrep{F,Y,AD,X,typeof(contexts)}(
        batch_size_settings, batched_seeds, batched_results, pullback_prep
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
    prep::PushforwardJacobianPrep{<:BatchSizeSettings{B,true,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,aligned,C}
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
    prep::PushforwardJacobianPrep{<:BatchSizeSettings{B,false,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,aligned,C}
    (; batch_size_settings, batched_seeds, pushforward_prep) = prep
    (; A, B_last) = batch_size_settings

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, backend, x, batched_seeds[1], contexts...
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
    prep::PullbackJacobianPrep{<:BatchSizeSettings{B,true,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,aligned,C}
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
    prep::PullbackJacobianPrep{<:BatchSizeSettings{B,false,aligned}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,aligned,C}
    (; batch_size_settings, batched_seeds, pullback_prep) = prep
    (; A, B_last) = batch_size_settings

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., prep.pullback_prep, backend, x, batched_seeds[1], contexts...
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
    prep::PushforwardJacobianPrep{<:BatchSizeSettings{B}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (; batch_size_settings, batched_seeds, batched_results, pushforward_prep) = prep
    (; N) = batch_size_settings

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, backend, x, batched_seeds[1], contexts...
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
    prep::PullbackJacobianPrep{<:BatchSizeSettings{B}},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (; batch_size_settings, batched_seeds, batched_results, pullback_prep) = prep
    (; N) = batch_size_settings

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_prep, backend, x, batched_seeds[1], contexts...
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
