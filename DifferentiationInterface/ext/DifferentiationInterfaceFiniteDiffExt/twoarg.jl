## Pushforward

struct FiniteDiffTwoArgPushforwardPrep{SIG,C,R,A,D} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f!,
    y,
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    cache = if x isa Number
        nothing
    else
        JVPCache(similar(x), similar(y), fdtype(backend))
    end
    relstep = if isnothing(backend.relstep)
        default_relstep(fdtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep = if isnothing(backend.absstep)
        relstep
    else
        backend.relstep
    end
    dir = backend.dir
    return FiniteDiffTwoArgPushforwardPrep(_sig, cache, relstep, absstep, dir)
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::FiniteDiffTwoArgPushforwardPrep{SIG,Nothing},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {SIG,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    function step(t::Number, dx)
        new_y = similar(y)
        f!(new_y, x .+ t .* dx, map(DI.unwrap, contexts)...)
        return new_y
    end
    ty = map(tx) do dx
        finite_difference_derivative(
            Base.Fix2(step, dx),
            zero(eltype(x)),
            fdtype(backend),
            eltype(y),
            y;
            relstep,
            absstep,
            dir,
        )
    end
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, ty
end

function DI.pushforward(
    f!,
    y,
    prep::FiniteDiffTwoArgPushforwardPrep{SIG,<:JVPCache},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {SIG,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    ty = map(tx) do dx
        dy = similar(y)
        finite_difference_jvp!(dy, fc!, x, dx, prep.cache; relstep, absstep, dir)
        dy
    end
    return ty
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::FiniteDiffTwoArgPushforwardPrep{SIG,<:JVPCache},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {SIG,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    ty = map(tx) do dx
        dy = similar(y)
        finite_difference_jvp!(dy, fc!, x, dx, prep.cache; relstep, absstep, dir)
        dy
    end
    fc!(y, x)
    return y, ty
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::FiniteDiffTwoArgPushforwardPrep{SIG,<:JVPCache},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {SIG,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        finite_difference_jvp!(dy, fc!, x, dx, prep.cache; relstep, absstep, dir)
    end
    return ty
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::FiniteDiffTwoArgPushforwardPrep{SIG,<:JVPCache},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {SIG,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        finite_difference_jvp!(dy, fc!, x, dx, prep.cache; relstep, absstep, dir)
    end
    fc!(y, x)
    return y, ty
end

## Derivative

struct FiniteDiffTwoArgDerivativePrep{SIG,C,R,A,D} <: DI.DerivativePrep{SIG}
    _sig::Val{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_derivative_nokwarg(
    strict::Val, f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    df = similar(y)
    cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    relstep = if isnothing(backend.relstep)
        default_relstep(fdtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep = if isnothing(backend.absstep)
        relstep
    else
        backend.relstep
    end
    dir = backend.dir
    return FiniteDiffTwoArgDerivativePrep(_sig, cache, relstep, absstep, dir)
end

function DI.prepare!_derivative(
    f!,
    y,
    old_prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    if y isa Vector
        (; cache) = old_prep
        cache.fx isa Union{Number,Nothing} || resize!(cache.fx, length(y))
        cache.c1 isa Union{Number,Nothing} || resize!(cache.c1, length(y))
        cache.c2 isa Union{Number,Nothing} || resize!(cache.c2, length(y))
        cache.c3 isa Union{Number,Nothing} || resize!(cache.c3, length(y))
        return old_prep
    else
        return DI.prepare_derivative_nokwarg(
            DI.is_strict(old_prep), f!, y, backend, x, contexts...
        )
    end
end

function DI.value_and_derivative(
    f!,
    y,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(y, x)
    der = finite_difference_gradient(fc!, x, prep.cache; relstep, absstep, dir)
    return y, der
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(y, x)
    finite_difference_gradient!(der, fc!, x, prep.cache; relstep, absstep, dir)
    return y, der
end

function DI.derivative(
    f!,
    y,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(y, x)
    der = finite_difference_gradient(fc!, x, prep.cache; relstep, absstep, dir)
    return der
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::FiniteDiffTwoArgDerivativePrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    finite_difference_gradient!(der, fc!, x, prep.cache; relstep, absstep, dir)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianPrep{SIG,C,R,A,D} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_jacobian_nokwarg(
    strict::Val, f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    relstep = if isnothing(backend.relstep)
        default_relstep(fdjtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep = if isnothing(backend.absstep)
        relstep
    else
        backend.relstep
    end
    dir = backend.dir
    return FiniteDiffTwoArgJacobianPrep(_sig, cache, relstep, absstep, dir)
end

function DI.prepare!_jacobian(
    f!,
    y,
    old_prep::FiniteDiffTwoArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    if x isa Vector && y isa Vector
        (; cache) = old_prep
        cache.x1 isa Union{Number,Nothing} || resize!(cache.x1, length(x))
        cache.x2 isa Union{Number,Nothing} || resize!(cache.x2, length(x))
        cache.fx isa Union{Number,Nothing} || resize!(cache.fx, length(y))
        cache.fx1 isa Union{Number,Nothing} || resize!(cache.fx1, length(y))
        cache.colorvec = 1:length(x)
        cache.sparsity = nothing
        return old_prep
    else
        return DI.prepare_jacobian_nokwarg(
            DI.is_strict(old_prep), f!, y, backend, x, contexts...
        )
    end
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::FiniteDiffTwoArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::FiniteDiffTwoArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    prep::FiniteDiffTwoArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::FiniteDiffTwoArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    return jac
end
