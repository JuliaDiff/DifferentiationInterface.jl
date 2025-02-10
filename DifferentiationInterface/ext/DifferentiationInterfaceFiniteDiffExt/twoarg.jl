## Pushforward

struct FiniteDiffTwoArgPushforwardPrep{C,R,A,D} <: DI.PushforwardPrep
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_pushforward(
    f!, y, backend::AutoFiniteDiff, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
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
    return FiniteDiffTwoArgPushforwardPrep(cache, relstep, absstep, dir)
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::FiniteDiffTwoArgPushforwardPrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    prep::FiniteDiffTwoArgPushforwardPrep{<:JVPCache},
    ::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
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
    prep::FiniteDiffTwoArgPushforwardPrep{<:JVPCache},
    ::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
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
    prep::FiniteDiffTwoArgPushforwardPrep{<:JVPCache},
    ::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
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
    prep::FiniteDiffTwoArgPushforwardPrep{<:JVPCache},
    ::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        finite_difference_jvp!(dy, fc!, x, dx, prep.cache; relstep, absstep, dir)
    end
    fc!(y, x)
    return y, ty
end

## Derivative

struct FiniteDiffTwoArgDerivativePrep{C,R,A,D} <: DI.DerivativePrep
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_derivative(
    f!, y, backend::AutoFiniteDiff, x, ::Vararg{DI.Context,C}
) where {C}
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
    return FiniteDiffTwoArgDerivativePrep(cache, relstep, absstep, dir)
end

function DI.value_and_derivative(
    f!,
    y,
    prep::FiniteDiffTwoArgDerivativePrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    fc!(y, x)
    der = finite_difference_gradient(fc!, x, prep.cache; relstep, absstep, dir)
    return y, der
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::FiniteDiffTwoArgDerivativePrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    fc!(y, x)
    finite_difference_gradient!(der, fc!, x, prep.cache; relstep, absstep, dir)
    return y, der
end

function DI.derivative(
    f!,
    y,
    prep::FiniteDiffTwoArgDerivativePrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    fc!(y, x)
    der = finite_difference_gradient(fc!, x, prep.cache; relstep, absstep, dir)
    return der
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::FiniteDiffTwoArgDerivativePrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    finite_difference_gradient!(der, fc!, x, prep.cache; relstep, absstep, dir)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianPrep{C,R,A,D} <: DI.JacobianPrep
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_jacobian(
    f!, y, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
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
    return FiniteDiffTwoArgJacobianPrep(cache, relstep, absstep, dir)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
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
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    fc!(y, x)
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::FiniteDiffTwoArgJacobianPrep,
    ::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    (; relstep, absstep, dir) = prep
    fc! = DI.with_contexts(f!, contexts...)
    finite_difference_jacobian!(jac, fc!, x, prep.cache; relstep, absstep, dir)
    return jac
end
