## Pushforward

struct FiniteDiffOneArgPushforwardPrep{SIG,C,R,A,D} <: DI.PushforwardPrep{SIG}
    _sig::Type{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_pushforward(
    f,
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {C}
    SIG = DI.signature(f, backend, x, tx, contexts...; strict)
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    cache = if x isa Number || y isa Number
        nothing
    else
        JVPCache(similar(x), y, fdtype(backend))
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
    return FiniteDiffOneArgPushforwardPrep(SIG, cache, relstep, absstep, dir)
end

function DI.pushforward(
    f,
    prep::FiniteDiffOneArgPushforwardPrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    step(t::Number, dx) = f(x .+ t .* dx, map(DI.unwrap, contexts)...)
    ty = map(tx) do dx
        finite_difference_derivative(
            Base.Fix2(step, dx), zero(eltype(x)), fdtype(backend); relstep, absstep, dir
        )
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::FiniteDiffOneArgPushforwardPrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    step(t::Number, dx) = f(x .+ t .* dx, map(DI.unwrap, contexts)...)
    y = f(x, map(DI.unwrap, contexts)...)
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
    return y, ty
end

function DI.pushforward(
    f,
    prep::FiniteDiffOneArgPushforwardPrep{<:JVPCache},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    ty = map(tx) do dx
        finite_difference_jvp(fc, x, dx, prep.cache; relstep, absstep, dir)
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::FiniteDiffOneArgPushforwardPrep{<:JVPCache},
    backend::AutoFiniteDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    ty = map(tx) do dx
        finite_difference_jvp(fc, x, dx, prep.cache, y; relstep, absstep, dir)
    end
    return y, ty
end

## Derivative

struct FiniteDiffOneArgDerivativePrep{SIG,C,R,A,D} <: DI.DerivativePrep{SIG}
    _sig::Type{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_derivative(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}; strict::Bool=false
) where {C}
    SIG = DI.signature(f, backend, x, contexts...; strict)
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    cache = if y isa Number
        nothing
    elseif y isa AbstractArray
        df = similar(y)
        cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_NOT_INPLACE)
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
    return FiniteDiffOneArgDerivativePrep(SIG, cache, relstep, absstep, dir)
end

### Scalar to scalar

function DI.derivative(
    f,
    prep::FiniteDiffOneArgDerivativePrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_derivative(fc, x, fdtype(backend); relstep, absstep, dir)
end

function DI.value_and_derivative(
    f,
    prep::FiniteDiffOneArgDerivativePrep{Nothing},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    return (
        y,
        finite_difference_derivative(
            fc, x, fdtype(backend), eltype(y), y; relstep, absstep, dir
        ),
    )
end

### Scalar to array

function DI.derivative(
    f,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_gradient(fc, x, prep.cache; relstep, absstep, dir)
end

function DI.derivative!(
    f,
    der,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_gradient!(der, fc, x, prep.cache; relstep, absstep, dir)
end

function DI.value_and_derivative(
    f,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.with_contexts(f, contexts...)
    (; relstep, absstep, dir) = prep
    y = fc(x)
    return (y, finite_difference_gradient(fc, x, prep.cache; relstep, absstep, dir))
end

function DI.value_and_derivative!(
    f,
    der,
    prep::FiniteDiffOneArgDerivativePrep{<:GradientCache},
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return (
        fc(x), finite_difference_gradient!(der, fc, x, prep.cache; relstep, absstep, dir)
    )
end

## Gradient

struct FiniteDiffGradientPrep{SIG,C,R,A,D} <: DI.GradientPrep{SIG}
    _sig::Type{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_gradient(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}; strict::Bool=false
) where {C}
    SIG = DI.signature(f, backend, x, contexts...; strict)
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    df = zero(y) .* x
    cache = GradientCache(df, x, fdtype(backend))
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
    return FiniteDiffGradientPrep(SIG, cache, relstep, absstep, dir)
end

function DI.gradient(
    f,
    prep::FiniteDiffGradientPrep,
    backend::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_gradient(fc, x, prep.cache; relstep, absstep, dir)
end

function DI.value_and_gradient(
    f,
    prep::FiniteDiffGradientPrep,
    backend::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return fc(x), finite_difference_gradient(fc, x, prep.cache; relstep, absstep, dir)
end

function DI.gradient!(
    f,
    grad,
    prep::FiniteDiffGradientPrep,
    backend::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_gradient!(grad, fc, x, prep.cache; relstep, absstep, dir)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::FiniteDiffGradientPrep,
    backend::AutoFiniteDiff,
    x::AbstractArray,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return (
        fc(x), finite_difference_gradient!(grad, fc, x, prep.cache; relstep, absstep, dir)
    )
end

## Jacobian

struct FiniteDiffOneArgJacobianPrep{SIG,C,R,A,D} <: DI.JacobianPrep{SIG}
    _sig::Type{SIG}
    cache::C
    relstep::R
    absstep::A
    dir::D
end

function DI.prepare_jacobian(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}; strict::Bool=false
) where {C}
    SIG = DI.signature(f, backend, x, contexts...; strict)
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
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
    return FiniteDiffOneArgJacobianPrep(SIG, cache, relstep, absstep, dir)
end

function DI.jacobian(
    f,
    prep::FiniteDiffOneArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_jacobian(fc, x, prep.cache; relstep, absstep, dir)
end

function DI.value_and_jacobian(
    f,
    prep::FiniteDiffOneArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.with_contexts(f, contexts...)
    (; relstep, absstep, dir) = prep
    y = fc(x)
    return (y, finite_difference_jacobian(fc, x, prep.cache, y; relstep, absstep, dir))
end

function DI.jacobian!(
    f,
    jac,
    prep::FiniteDiffOneArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    return copyto!(
        jac,
        finite_difference_jacobian(
            fc, x, prep.cache; jac_prototype=jac, relstep, absstep, dir
        ),
    )
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::FiniteDiffOneArgJacobianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep, absstep, dir) = prep
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    return (
        y,
        copyto!(
            jac,
            finite_difference_jacobian(
                fc, x, prep.cache, y; jac_prototype=jac, relstep, absstep, dir
            ),
        ),
    )
end

## Hessian

struct FiniteDiffHessianPrep{SIG,C1,C2,RG,AG,RH,AH} <: DI.HessianPrep{SIG}
    _sig::Type{SIG}
    gradient_cache::C1
    hessian_cache::C2
    relstep_g::RG
    absstep_g::AG
    relstep_h::RH
    absstep_h::AH
end

function DI.prepare_hessian(
    f, backend::AutoFiniteDiff, x, contexts::Vararg{DI.Context,C}; strict::Bool=false
) where {C}
    SIG = DI.signature(f, backend, x, contexts...; strict)
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    df = zero(y) .* x
    gradient_cache = GradientCache(df, x, fdtype(backend))
    hessian_cache = HessianCache(x, fdhtype(backend))
    relstep_g = if isnothing(backend.relstep)
        default_relstep(fdtype(backend), eltype(x))
    else
        backend.relstep
    end
    relstep_h = if isnothing(backend.relstep)
        default_relstep(fdhtype(backend), eltype(x))
    else
        backend.relstep
    end
    absstep_g = isnothing(backend.absstep) ? relstep_g : backend.absstep
    absstep_h = isnothing(backend.absstep) ? relstep_h : backend.absstep
    return FiniteDiffHessianPrep(
        SIG, gradient_cache, hessian_cache, relstep_g, absstep_g, relstep_h, absstep_h
    )
end

function DI.hessian(
    f,
    prep::FiniteDiffHessianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep_h, absstep_h) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_hessian(
        fc, x, prep.hessian_cache; relstep=relstep_h, absstep=absstep_h
    )
end

function DI.hessian!(
    f,
    hess,
    prep::FiniteDiffHessianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep_h, absstep_h) = prep
    fc = DI.with_contexts(f, contexts...)
    return finite_difference_hessian!(
        hess, fc, x, prep.hessian_cache; relstep=relstep_h, absstep=absstep_h
    )
end

function DI.value_gradient_and_hessian(
    f,
    prep::FiniteDiffHessianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep_g, absstep_g, relstep_h, absstep_h) = prep
    fc = DI.with_contexts(f, contexts...)
    grad = finite_difference_gradient(
        fc, x, prep.gradient_cache; relstep=relstep_g, absstep=absstep_g
    )
    hess = finite_difference_hessian(
        fc, x, prep.hessian_cache; relstep=relstep_h, absstep=absstep_h
    )
    return fc(x), grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::FiniteDiffHessianPrep,
    backend::AutoFiniteDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; relstep_g, absstep_g, relstep_h, absstep_h) = prep
    fc = DI.with_contexts(f, contexts...)
    finite_difference_gradient!(
        grad, fc, x, prep.gradient_cache; relstep=relstep_g, absstep=absstep_g
    )
    finite_difference_hessian!(
        hess, fc, x, prep.hessian_cache; relstep=relstep_h, absstep=absstep_h
    )
    return fc(x), grad, hess
end
