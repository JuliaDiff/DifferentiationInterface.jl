## Pushforward

struct ForwardDiffTwoArgPushforwardPrep{SIG,T,X,Y,CD} <: DI.PushforwardPrep{SIG}
    xdual_tmp::X
    ydual_tmp::Y
    contexts_dual::CD
end

function DI.prepare_pushforward(
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {F,B,C}
    SIG = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    T = tag_type(f!, backend, x)
    xdual_tmp = make_dual_similar(T, x, tx)
    ydual_tmp = make_dual_similar(T, y, tx)  # tx only for batch size
    contexts_dual = translate_toprep(eltype(xdual_tmp), contexts)
    return ForwardDiffTwoArgPushforwardPrep{
        SIG,T,typeof(xdual_tmp),typeof(ydual_tmp),typeof(contexts_dual)
    }(
        xdual_tmp, ydual_tmp, contexts_dual
    )
end

function compute_ydual_twoarg(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    x::Number,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    (; ydual_tmp) = prep
    xdual_tmp = make_dual(T, x, tx)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    f!(ydual_tmp, xdual_tmp, contexts_dual...)
    return ydual_tmp
end

function compute_ydual_twoarg(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    (; xdual_tmp, ydual_tmp) = prep
    make_dual!(T, xdual_tmp, x, tx)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    f!(ydual_tmp, xdual_tmp, contexts_dual...)
    return ydual_tmp
end

function DI.value_and_pushforward(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
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
    backend::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,T,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    myvalue!(T, y, ydual_tmp)
    mypartials!(T, ty, ydual_tmp)
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    backend::AutoForwardDiff,
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,T,B,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual_tmp)
    return ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::ForwardDiffTwoArgPushforwardPrep{T},
    backend::AutoForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,T,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual_tmp = compute_ydual_twoarg(f!, y, prep, x, tx, contexts...)
    mypartials!(T, ty, ydual_tmp)
    return ty
end

## Derivative

### Unprepared, only when tag is not specified

function DI.value_and_derivative(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (T === Nothing && contexts isa NTuple{C,DI.GeneralizedConstant})
        fc! = DI.with_contexts(f!, contexts...)
        result = MutableDiffResult(y, (similar(y),))
        result = derivative!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.derivative(result)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.value_and_derivative(f!, y, prep, backend, x, contexts...)
    end
end

function DI.value_and_derivative!(
    f!::F, y, der, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (T === Nothing && contexts isa NTuple{C,DI.GeneralizedConstant})
        fc! = DI.with_contexts(f!, contexts...)
        result = MutableDiffResult(y, (der,))
        result = derivative!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.derivative(result)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.value_and_derivative!(f!, y, der, prep, backend, x, contexts...)
    end
end

function DI.derivative(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (T === Nothing && contexts isa NTuple{C,DI.GeneralizedConstant})
        fc! = DI.with_contexts(f!, contexts...)
        return derivative(fc!, y, x)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.derivative(f!, y, prep, backend, x, contexts...)
    end
end

function DI.derivative!(
    f!::F, y, der, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (T === Nothing && contexts isa NTuple{C,DI.GeneralizedConstant})
        fc! = DI.with_contexts(f!, contexts...)
        return derivative!(der, fc!, y, x)
    else
        prep = DI.prepare_derivative(f!, y, backend, x, contexts...)
        return DI.derivative!(f!, y, der, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffTwoArgDerivativePrep{SIG,C,CD} <: DI.DerivativePrep{SIG}
    _sig::Type{SIG}
    config::C
    contexts_dual::CD
end

function DI.prepare_derivative(
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {F,C}
    SIG = DI.signature(f!, y, backend, x, contexts...; strict)
    tag = get_tag(f!, backend, x)
    config = DerivativeConfig(nothing, y, x, tag)
    contexts_dual = translate_toprep(dual_type(config), contexts)
    return ForwardDiffTwoArgDerivativePrep(SIG, config, contexts_dual)
end

function DI.prepare!_derivative(
    f!::F,
    y,
    old_prep::ForwardDiffTwoArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {F,C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    if y isa Vector
        (; config) = old_prep
        resize!(config.duals, length(y))
        return old_prep
    else
        return DI.prepare_derivative(
            f!, y, backend, x, contexts...; strict=DI.is_strict(old_prep)
        )
    end
end

function DI.value_and_derivative(
    f!::F,
    y,
    prep::ForwardDiffTwoArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    result = MutableDiffResult(y, (similar(y),))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    result = derivative!(result, fc!, y, x, prep.config, Val(false))
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_derivative!(
    f!::F,
    y,
    der,
    prep::ForwardDiffTwoArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    result = MutableDiffResult(y, (der,))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    result = derivative!(result, fc!, y, x, prep.config, Val(false))
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.derivative(
    f!::F,
    y,
    prep::ForwardDiffTwoArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    return derivative(fc!, y, x, prep.config, Val(false))
end

function DI.derivative!(
    f!::F,
    y,
    der,
    prep::ForwardDiffTwoArgDerivativePrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    return derivative!(der, fc!, y, x, prep.config, Val(false))
end

## Jacobian

### Unprepared, only when chunk size and tag are not specified

function DI.value_and_jacobian(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.GeneralizedConstant}
    )
        fc! = DI.with_contexts(f!, contexts...)
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
    f!::F, y, jac, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.GeneralizedConstant}
    )
        fc! = DI.with_contexts(f!, contexts...)
        result = MutableDiffResult(y, (jac,))
        result = jacobian!(result, fc!, y, x)
        return DiffResults.value(result), DiffResults.jacobian(result)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.value_and_jacobian!(f!, y, jac, prep, backend, x, contexts...)
    end
end

function DI.jacobian(
    f!::F, y, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.GeneralizedConstant}
    )
        fc! = DI.with_contexts(f!, contexts...)
        return jacobian(fc!, y, x)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.jacobian(f!, y, prep, backend, x, contexts...)
    end
end

function DI.jacobian!(
    f!::F, y, jac, backend::AutoForwardDiff{chunksize,T}, x, contexts::Vararg{DI.Context,C}
) where {F,C,chunksize,T}
    if (
        isnothing(chunksize) &&
        T === Nothing &&
        contexts isa NTuple{C,DI.GeneralizedConstant}
    )
        fc! = DI.with_contexts(f!, contexts...)
        return jacobian!(jac, fc!, y, x)
    else
        prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
        return DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
    end
end

### Prepared

struct ForwardDiffTwoArgJacobianPrep{SIG,C,CD} <: DI.JacobianPrep{SIG}
    _sig::Type{SIG}
    config::C
    contexts_dual::CD
end

function DI.prepare_jacobian(
    f!::F,
    y,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {F,C}
    SIG = DI.signature(f!, y, backend, x, contexts...; strict)
    chunk = choose_chunk(backend, x)
    tag = get_tag(f!, backend, x)
    config = JacobianConfig(nothing, y, x, chunk, tag)
    contexts_dual = translate_toprep(dual_type(config), contexts)
    return ForwardDiffTwoArgJacobianPrep(SIG, config, contexts_dual)
end

function DI.prepare!_jacobian(
    f!::F,
    y,
    old_prep::ForwardDiffTwoArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {F,C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    if x isa Vector && y isa Vector
        (; config) = old_prep
        (yduals, xduals) = config.duals
        resize!(yduals, length(y))
        resize!(xduals, length(x))
        return old_prep
    else
        return DI.prepare_jacobian(
            f!, y, backend, x, contexts...; strict=DI.is_strict(old_prep)
        )
    end
end

function DI.value_and_jacobian(
    f!::F,
    y,
    prep::ForwardDiffTwoArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    result = jacobian!(result, fc!, y, x, prep.config, Val(false))
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.value_and_jacobian!(
    f!::F,
    y,
    jac,
    prep::ForwardDiffTwoArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    result = MutableDiffResult(y, (jac,))
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    result = jacobian!(result, fc!, y, x, prep.config, Val(false))
    return DiffResults.value(result), DiffResults.jacobian(result)
end

function DI.jacobian(
    f!::F,
    y,
    prep::ForwardDiffTwoArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    return jacobian(fc!, y, x, prep.config, Val(false))
end

function DI.jacobian!(
    f!::F,
    y,
    jac,
    prep::ForwardDiffTwoArgJacobianPrep,
    backend::AutoForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, old_prep, backend, x, contexts...)
    contexts_dual = translate_prepared(contexts, prep.contexts_dual)
    fc! = DI.FixTail(f!, contexts_dual...)
    CHK = tag_type(backend) === Nothing
    if CHK
        checktag(prep.config, f!, x)
    end
    return jacobian!(jac, fc!, y, x, prep.config, Val(false))
end
