## Pushforward

struct ForwardDiffTwoArgPushforwardPrep{T,X,Y} <: PushforwardPrep
    xdual_tmp::X
    ydual_tmp::Y
end

function DI.prepare_pushforward(
    f!::F, y, backend::AutoForwardDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {F,C}
    T = tag_type(f!, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    ydual_tmp = make_dual_similar(T, y, tx)  # dx only for batch size
    return ForwardDiffTwoArgPushforwardPrep{T,typeof(xdual_tmp),typeof(ydual_tmp)}(
        xdual_tmp, ydual_tmp
    )
end

function compute_ydual_twoarg(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    x::Number,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    (; ydual_tmp) = prep
    xdual_tmp = make_dual(T, x, tx)
    contexts_dual = translate(T, Val(B), contexts...)
    f!(ydual_tmp, xdual_tmp, contexts_dual...)
    return ydual_tmp
end

function compute_ydual_twoarg(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    (; xdual_tmp, ydual_tmp) = prep
    make_dual!(T, xdual_tmp, x, tx)
    contexts_dual = translate(T, Val(B), contexts...)
    f!(ydual_tmp, xdual_tmp, contexts_dual...)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    myvalue!(T, y, ydual_tmp)
    ty = mypartials(T, Val(B), ydual_tmp)
    return y, ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    myvalue!(T, y, ydual_tmp)
    mypartials!(T, ty, ydual_tmp)
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,T,B,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual_tmp)
    return ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    ::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,T,C}
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    mypartials!(T, ty, ydual_tmp)
    return ty
end

## Derivative

### Unprepared, only when tag is not specified

function DI.value_and_derivative(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if T === Nothing
        fc! = with_contexts(f!, contexts...)
        result = MutableDiffResult(y, (similar(y),))
        result = derivative!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.derivative(result)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.value_and_derivative(f!, y, prep, backend, x, contexts...)
    end
end

function DI.value_and_derivative!(
    f!::F, y, der, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if T === Nothing
        fc! = with_contexts(f!, contexts...)
        result = MutableDiffResult(y, (der,))
        result = derivative!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.derivative(result)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.value_and_derivative!(f!, y, der, prep, backend, x, contexts...)
    end
end

function DI.derivative(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if T === Nothing
        fc! = with_contexts(f!, contexts...)
        return derivative(fc!, y, x)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.derivative(f!, y, prep, backend, x, contexts...)
    end
end

function DI.derivative!(
    f!::F, y, der, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if T === Nothing
        fc! = with_contexts(f!, contexts...)
        return derivative!(der, fc!, y, x)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.derivative!(f!, y, der, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffTwoArgDerivativePrep{C} <: DerivativePrep
    config::C
end

function DI.prepare_derivative(
    f!::F, y, backend::AutoForwardDiff, x, contexts::Vararg{Constant,C}
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    tag = get_tag(fc!, backend, x)
    config = DerivativeConfig(fc!, y, x, tag)
    return ForwardDiffTwoArgDerivativePrep(config)
end

function DI.value_and_derivative(
    f!::F,
    y,
    prep::ForwardDiffTwoArgDerivativePrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    result = MutableDiffResult(y, (similar(y),))
    result = derivative!(result, fc!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!::F,
    y,
    der,
    prep::ForwardDiffTwoArgDerivativePrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    result = MutableDiffResult(y, (der,))
    result = derivative!(result, fc!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!::F,
    y,
    prep::ForwardDiffTwoArgDerivativePrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    return derivative(fc!, y, x, prep.config)
end

function DI.derivative!(
    f!::F,
    y,
    der,
    prep::ForwardDiffTwoArgDerivativePrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    return derivative!(der, fc!, y, x, prep.config)
end

## Jacobian

### Unprepared, only when chunk size and tag are not specified

function DI.value_and_jacobian(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if isnothing(chunksize) && T === Nothing
        fc! = with_contexts(f!, contexts...)
        jac = similar(y, length(y), length(x))
        result = MutableDiffResult(y, (jac,))
        result = jacobian!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.jacobian(result)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.value_and_jacobian(f!, y, prep, backend, x, contexts...)
    end
end

function DI.value_and_jacobian!(
    f!::F, y, jac, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if isnothing(chunksize) && T === Nothing
        fc! = with_contexts(f!, contexts...)
        result = MutableDiffResult(y, (jac,))
        result = jacobian!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.jacobian(result)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.value_and_jacobian!(f!, y, jac, prep, backend, x, contexts...)
    end
end

function DI.jacobian(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if isnothing(chunksize) && T === Nothing
        fc! = with_contexts(f!, contexts...)
        return jacobian(fc!, y, x)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.jacobian(f!, y, prep, backend, x, contexts...)
    end
end

function DI.jacobian!(
    f!::F, y, jac, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{Constant,C}
) where {F,C,chunksize,T}
    if isnothing(chunksize) && T === Nothing
        fc! = with_contexts(f!, contexts...)
        return jacobian!(jac, fc!, y, x)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffTwoArgJacobianPrep{C} <: JacobianPrep
    config::C
end

function DI.prepare_jacobian(
    f!::F, y, backend::AutoForwardDiff, x, contexts::Vararg{Constant,C}
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    chunk = choose_chunk(backend, x)
    tag = get_tag(fc!, backend, x)
    config = JacobianConfig(fc!, y, x, chunk, tag)
    return ForwardDiffTwoArgJacobianPrep(config)
end

function DI.value_and_jacobian(
    f!::F,
    y,
    prep::ForwardDiffTwoArgJacobianPrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(
    f!::F,
    y,
    jac,
    prep::ForwardDiffTwoArgJacobianPrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!::F,
    y,
    prep::ForwardDiffTwoArgJacobianPrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    return jacobian(fc!, y, x, prep.config)
end

function DI.jacobian!(
    f!::F,
    y,
    jac,
    prep::ForwardDiffTwoArgJacobianPrep,
    ::AutoForwardDiff,
    x,
    contexts::Vararg{Constant,C},
) where {F,C}
    fc! = with_contexts(f!, contexts...)
    return jacobian!(jac, fc!, y, x, prep.config)
end
