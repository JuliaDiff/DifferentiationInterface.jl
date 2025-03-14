## Pushforward

### Unprepared (avoid working on `similar(x)`)

function DI.value_and_pushforward(
    f::F, backend::AutoForwardDiff, x, tx::NTuple{B}, contexts::Vararg{DI.Context,C}
) where {F,B,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, tx)
    contexts_dual = translate(eltype(xdual), contexts)
    ydual = f(xdual, contexts_dual...)
    y = myvalue(T, ydual)
    ty = mypartials(T, Val(B), ydual)
    return y, ty
end

function DI.value_and_pushforward!(
    f::F,
    ty::NTuple{B},
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, tx)
    contexts_dual = translate(eltype(xdual), contexts)
    ydual = f(xdual, contexts_dual...)
    y = myvalue(T, ydual)
    mypartials!(T, ty, ydual)
    return y, ty
end

function DI.pushforward(
    f::F, backend::AutoForwardDiff, x, tx::NTuple{B}, contexts::Vararg{DI.Context,C}
) where {F,B,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, tx)
    contexts_dual = translate(eltype(xdual), contexts)
    ydual = f(xdual, contexts_dual...)
    ty = mypartials(T, Val(B), ydual)
    return ty
end

function DI.pushforward!(
    f::F,
    ty::NTuple{B},
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, tx)
    contexts_dual = translate(eltype(xdual), contexts)
    ydual = f(xdual, contexts_dual...)
    mypartials!(T, ty, ydual)
    return ty
end

### Prepared

struct ForwardDiffOneArgPushforwardPrep{T,X,CD} <: DI.PushforwardPrep
    xdual_tmp::X
    contexts_dual::CD
end

function DI.prepare_pushforward(
    f::F, backend::AutoForwardDiff, x, tx::NTuple{B}, contexts::Vararg{DI.Context,C}
) where {F,B,C}
    T = tag_type(f, backend, x)
    if DI.ismutable_array(x)
        xdual_tmp = make_dual_similar(T, x, tx)
    else
        xdual_tmp = nothing
    end
    contexts_dual = translate_toprep(Dual{T,eltype(x),B}, contexts)
    return ForwardDiffOneArgPushforwardPrep{T,typeof(xdual_tmp),typeof(contexts_dual)}(
        xdual_tmp, contexts_dual
    )
end

function compute_ydual_onearg(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    x::Number,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    xdual = make_dual(T, x, tx)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    ydual = f(xdual, contexts_dual...)
    return ydual
end

function compute_ydual_onearg(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    if DI.ismutable_array(x)
        make_dual!(T, prep.xdual_tmp, x, tx)
        xdual_tmp = prep.xdual_tmp
    else
        xdual_tmp = make_dual(T, x, tx)
    end
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    ydual = f(xdual_tmp, contexts_dual...)
    return ydual
end

function DI.value_and_pushforward(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    y = myvalue(T, ydual)
    ty = mypartials(T, Val(B), ydual)
    return y, ty
end

function DI.value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,T,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    y = myvalue(T, ydual)
    mypartials!(T, ty, ydual)
    return y, ty
end

function DI.pushforward(
    f::F,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual)
    return ty
end

function DI.pushforward!(
    f::F,
    ty::NTuple,
    prep::ForwardDiffOneArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,T,C}
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    mypartials!(T, ty, ydual)
    return ty
end

## Derivative

struct ForwardDiffOneArgDerivativePrep{E} <: DI.DerivativePrep
    pushforward_prep::E
end

### Unprepared

function DI.value_and_derivative(
    f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    y, ty = DI.value_and_pushforward(f, backend, x, (one(x),), contexts...)
    return y, only(ty)
end

function DI.value_and_derivative!(
    f::F, der, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    y, _ = DI.value_and_pushforward!(f, (der,), backend, x, (one(x),), contexts...)
    return y, der
end

function DI.derivative(
    f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    return only(DI.pushforward(f, backend, x, (one(x),), contexts...))
end

function DI.derivative!(
    f::F, der, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    DI.pushforward!(f, (der,), backend, x, (one(x),), contexts...)
    return der
end

### Prepared

function DI.prepare_derivative(
    f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    pushforward_prep = DI.prepare_pushforward(f, backend, x, (one(x),), contexts...)
    return ForwardDiffOneArgDerivativePrep(pushforward_prep)
end

function DI.value_and_derivative(
    f::F,
    prep::ForwardDiffOneArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    y, ty = DI.value_and_pushforward(
        f, prep.pushforward_prep, backend, x, (one(x),), contexts...
    )
    return y, only(ty)
end

function DI.value_and_derivative!(
    f::F,
    der,
    prep::ForwardDiffOneArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    y, _ = DI.value_and_pushforward!(
        f, (der,), prep.pushforward_prep, backend, x, (one(x),), contexts...
    )
    return y, der
end

function DI.derivative(
    f::F,
    prep::ForwardDiffOneArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return only(
        DI.pushforward(f, prep.pushforward_prep, backend, x, (one(x),), contexts...)
    )
end

function DI.derivative!(
    f::F,
    der,
    prep::ForwardDiffOneArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.pushforward!(f, (der,), prep.pushforward_prep, backend, x, (one(x),), contexts...)
    return der
end

## Gradient

### Unprepared, only when chunk size and tag are not specified

function DI.value_and_gradient!(
    f::F, grad, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        result = DiffResult(zero(eltype(x)), (grad,))
        result = gradient!(result, fc, x)
        y = DR.value(result)
        grad === DR.gradient(result) || copyto!(grad, DR.gradient(result))
        return y, grad
    else
        prep = DI.prepare_gradient(f, backend, x, contexts...)
        return DI.value_and_gradient!(f, grad, prep, backend, x, contexts...)
    end
end

function DI.value_and_gradient(
    f::F, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        result = GradientResult(x)
        result = gradient!(result, fc, x)
        return DR.value(result), DR.gradient(result)
    else
        prep = DI.prepare_gradient(f, backend, x, contexts...)
        return DI.value_and_gradient(f, prep, backend, x, contexts...)
    end
end

function DI.gradient!(
    f::F, grad, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return gradient!(grad, fc, x)
    else
        prep = DI.prepare_gradient(f, backend, x, contexts...)
        return DI.gradient!(f, grad, prep, backend, x, contexts...)
    end
end

function DI.gradient(
    f::F, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return gradient(fc, x)
    else
        prep = DI.prepare_gradient(f, backend, x, contexts...)
        return DI.gradient(f, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffGradientPrep{C,CD} <: DI.GradientPrep
    config::C
    contexts_dual::CD
end

function DI.prepare_gradient(
    f::F, backend::AutoForwardDiff, x::AbstractArray, contexts::Vararg{DI.Context,C}
) where {F,C}
    chunk = choose_chunk(backend, x)
    tag = get_tag(f, backend, x)
    config = GradientConfig(nothing, x, chunk, tag)
    contexts_dual = translate_toprep(dual_type(config), contexts)
    return ForwardDiffGradientPrep(config, contexts_dual)
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::ForwardDiffGradientPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    result = DiffResult(zero(eltype(x)), (grad,))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    result = gradient!(result, fc, x, prep.config, Val(false))
    y = DR.value(result)
    grad === DR.gradient(result) || copyto!(grad, DR.gradient(result))
    return y, grad
end

function DI.value_and_gradient(
    f::F,
    prep::ForwardDiffGradientPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    result = GradientResult(x)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    result = gradient!(result, fc, x, prep.config, Val(false))
    return DR.value(result), DR.gradient(result)
end

function DI.gradient!(
    f::F,
    grad,
    prep::ForwardDiffGradientPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    return gradient!(grad, fc, x, prep.config, Val(false))
end

function DI.gradient(
    f::F,
    prep::ForwardDiffGradientPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    return gradient(fc, x, prep.config, Val(false))
end

## Jacobian

### Unprepared, only when chunk size and tag are not specified

function DI.value_and_jacobian!(
    f::F, jac, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        y = fc(x)
        result = DiffResult(y, (jac,))
        result = jacobian!(result, fc, x)
        y = DR.value(result)
        jac === DR.jacobian(result) || copyto!(jac, DR.jacobian(result))
        return y, jac
    else
        prep = DI.prepare_jacobian(f, backend, x, contexts...)
        return DI.value_and_jacobian!(f, jac, prep, backend, x, contexts...)
    end
end

function DI.value_and_jacobian(
    f::F, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return fc(x), jacobian(fc, x)
    else
        prep = DI.prepare_jacobian(f, backend, x, contexts...)
        return DI.value_and_jacobian(f, prep, backend, x, contexts...)
    end
end

function DI.jacobian!(
    f::F, jac, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return jacobian!(jac, fc, x)
    else
        prep = DI.prepare_jacobian(f, backend, x, contexts...)
        return DI.jacobian!(f, jac, prep, backend, x, contexts...)
    end
end

function DI.jacobian(
    f::F, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return jacobian(fc, x)
    else
        prep = DI.prepare_jacobian(f, backend, x, contexts...)
        return DI.jacobian(f, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffOneArgJacobianPrep{C,CD} <: DI.JacobianPrep
    config::C
    contexts_dual::CD
end

function DI.prepare_jacobian(
    f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    chunk = choose_chunk(backend, x)
    tag = get_tag(f, backend, x)
    config = JacobianConfig(nothing, x, chunk, tag)
    contexts_dual = translate_toprep(dual_type(config), contexts)
    return ForwardDiffOneArgJacobianPrep(config, contexts_dual)
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    prep::ForwardDiffOneArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    y = fc(x)
    result = DiffResult(y, (jac,))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    result = jacobian!(result, fc, x, prep.config, Val(false))
    y = DR.value(result)
    jac === DR.jacobian(result) || copyto!(jac, DR.jacobian(result))
    return y, jac
end

function DI.value_and_jacobian(
    f::F,
    prep::ForwardDiffOneArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    return fc(x), jacobian(fc, x, prep.config, Val(false))
end

function DI.jacobian!(
    f::F,
    jac,
    prep::ForwardDiffOneArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    return jacobian!(jac, fc, x, prep.config, Val(false))
end

function DI.jacobian(
    f::F,
    prep::ForwardDiffOneArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f, x)
    end
    return jacobian(fc, x, prep.config, Val(false))
end

## Second derivative

function DI.prepare_second_derivative(
    f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    return DI.NoSecondDerivativePrep()
end

function DI.second_derivative(
    f::F,
    ::DI.NoSecondDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    xdual2 = make_dual(T2, xdual, one(xdual))
    contexts_dual = translate(typeof(xdual2), contexts)
    ydual = f(xdual2, contexts_dual...)
    return myderivative(T, myderivative(T2, ydual))
end

function DI.second_derivative!(
    f::F,
    der2,
    ::DI.NoSecondDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    xdual2 = make_dual(T2, xdual, one(xdual))
    contexts_dual = translate(typeof(xdual2), contexts)
    ydual = f(xdual2, contexts_dual...)
    return myderivative!(T, der2, myderivative(T2, ydual))
end

function DI.value_derivative_and_second_derivative(
    f::F,
    ::DI.NoSecondDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    xdual2 = make_dual(T2, xdual, one(xdual))
    contexts_dual = translate(typeof(xdual2), contexts)
    ydual = f(xdual2, contexts_dual...)
    y = myvalue(T, myvalue(T2, ydual))
    der = myderivative(T, myvalue(T2, ydual))
    der2 = myderivative(T, myderivative(T2, ydual))
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    ::DI.NoSecondDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    T = tag_type(f, backend, x)
    xdual = make_dual(T, x, one(x))
    T2 = tag_type(f, backend, xdual)
    xdual2 = make_dual(T2, xdual, one(xdual))
    contexts_dual = translate(typeof(xdual2), contexts)
    ydual = f(xdual2, contexts_dual...)
    y = myvalue(T, myvalue(T2, ydual))
    myderivative!(T, der, myvalue(T2, ydual))
    myderivative!(T, der2, myderivative(T2, ydual))
    return y, der, der2
end

## HVP

#=
function DI.prepare_hvp(
    f::F, backend::AutoForwardDiff, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {F,C}
    return DI.prepare_hvp(f, DI.SecondOrder(backend, backend), x, tx, contexts...)
end

function DI.hvp(
    f::F,
    prep::DI.HVPPrep,
    backend::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.hvp(f, prep, DI.SecondOrder(backend, backend), x, tx, contexts...)
end

function DI.hvp!(
    f::F,
    tg::NTuple,
    prep::DI.HVPPrep,
    backend::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.hvp!(f, tg, prep, DI.SecondOrder(backend, backend), x, tx, contexts...)
end

function DI.gradient_and_hvp(
    f::F,
    prep::DI.HVPPrep,
    backend::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.gradient_and_hvp(
        f, prep, DI.SecondOrder(backend, backend), x, tx, contexts...
    )
end

function DI.gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::DI.HVPPrep,
    backend::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.gradient_and_hvp!(
        f, grad, tg, prep, DI.SecondOrder(backend, backend), x, tx, contexts...
    )
end
=#

## Hessian

### Unprepared, only when chunk size and tag are not specified

function DI.hessian!(
    f::F, hess, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return hessian!(hess, fc, x)
    else
        prep = DI.prepare_hessian(f, backend, x, contexts...)
        return DI.hessian!(f, hess, prep, backend, x, contexts...)
    end
end

function DI.hessian(
    f::F, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        return hessian(fc, x)
    else
        prep = DI.prepare_hessian(f, backend, x, contexts...)
        return DI.hessian(f, prep, backend, x, contexts...)
    end
end

function DI.value_gradient_and_hessian!(
    f::F,
    grad,
    hess,
    backend::AutoForwardDiff{chunksize,T},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        result = DiffResult(one(eltype(x)), (grad, hess))
        result = hessian!(result, fc, x)
        y = DR.value(result)
        grad === DR.gradient(result) || copyto!(grad, DR.gradient(result))
        hess === DR.hessian(result) || copyto!(hess, DR.hessian(result))
        return (y, grad, hess)
    else
        prep = DI.prepare_hessian(f, backend, x, contexts...)
        return DI.value_gradient_and_hessian!(f, grad, hess, prep, backend, x, contexts...)
    end
end

function DI.value_gradient_and_hessian(
    f::F, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.ConstantOrFunctionOrBackend}
    )
        fc = DI.with_contexts(f, contexts...)
        result = HessianResult(x)
        result = hessian!(result, fc, x)
        return (DR.value(result), DR.gradient(result), DR.hessian(result))
    else
        prep = DI.prepare_hessian(f, backend, x, contexts...)
        return DI.value_gradient_and_hessian(f, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffHessianPrep{C1,C2,CD} <: DI.HessianPrep
    array_config::C1
    result_config::C2
    contexts_dual::CD
end

function DI.prepare_hessian(
    f::F, backend::AutoForwardDiff, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    chunk = choose_chunk(backend, x)
    tag = get_tag(f, backend, x)
    result = HessianResult(x)
    array_config = HessianConfig(nothing, x, chunk, tag)
    result_config = HessianConfig(nothing, result, x, chunk, tag)
    contexts_dual = translate_toprep(dual_type(array_config), contexts)
    return ForwardDiffHessianPrep(array_config, result_config, contexts_dual)
end

function DI.hessian!(
    f::F,
    hess,
    prep::ForwardDiffHessianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.array_config, f, x)
    end
    return hessian!(hess, fc, x, prep.array_config, Val(false))
end

function DI.hessian(
    f::F,
    prep::ForwardDiffHessianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.array_config, f, x)
    end
    return hessian(fc, x, prep.array_config, Val(false))
end

function DI.value_gradient_and_hessian!(
    f::F,
    grad,
    hess,
    prep::ForwardDiffHessianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    result = DiffResult(one(eltype(x)), (grad, hess))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.result_config, f, x)
    end
    result = hessian!(result, fc, x, prep.result_config, Val(false))
    y = DR.value(result)
    grad === DR.gradient(result) || copyto!(grad, DR.gradient(result))
    hess === DR.hessian(result) || copyto!(hess, DR.hessian(result))
    return (y, grad, hess)
end

function DI.value_gradient_and_hessian(
    f::F,
    prep::ForwardDiffHessianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc = DI.FixTail(f, contexts_dual...)
    result = HessianResult(x)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.result_config, f, x)
    end
    result = hessian!(result, fc, x, prep.result_config, Val(false))
    return (DR.value(result), DR.gradient(result), DR.hessian(result))
end
