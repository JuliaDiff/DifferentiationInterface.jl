## Pullback

function DI.prepare_pullback(
    f!, y, ::AutoReverseDiff, x, ty::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoPullbackPrep()
end

### Array in

function DI.value_and_pullback(
    f!,
    y,
    ::DI.NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    tx = map(ty) do dy
        gradient(Fix2(dotclosure, dy), x)
    end
    fc!(y, x)
    return y, tx
end

function DI.value_and_pullback!(
    f!,
    y,
    tx::NTuple,
    ::DI.NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        gradient!(dx, Fix2(dotclosure, dy), x)
    end
    fc!(y, x)
    return y, tx
end

function DI.pullback(
    f!,
    y,
    ::DI.NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    tx = map(ty) do dy
        gradient(Fix2(dotclosure, dy), x)
    end
    return tx
end

function DI.pullback!(
    f!,
    y,
    tx::NTuple,
    ::DI.NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    function dotclosure(x, dy)
        y_copy = similar(y, eltype(x))
        fc!(y_copy, x)
        return dot(y_copy, dy)
    end
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        gradient!(dx, Fix2(dotclosure, dy), x)
    end
    return tx
end

### Number in, not supported

function DI.value_and_pullback(
    f!,
    y,
    ::DI.NoPullbackPrep,
    backend::AutoReverseDiff,
    x::Number,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    x_array = [x]
    function f!_array(_y::AbstractArray, _x_array, args...)
        return f!(_y, only(_x_array), args...)
    end
    y, tx_array = DI.value_and_pullback(f!_array, y, backend, x_array, ty, contexts...)
    return y, only.(tx_array)
end

## Jacobian

### Without contexts

@kwdef struct ReverseDiffTwoArgJacobianPrep{C,T} <: DI.JacobianPrep
    config::C
    tape::T
end

function DI.prepare_jacobian(f!, y, ::AutoReverseDiff{compile}, x) where {compile}
    if compile
        tape = ReverseDiff.compile(JacobianTape(f!, y, x))
        return ReverseDiffTwoArgJacobianPrep(; config=nothing, tape=tape)
    else
        config = JacobianConfig(y, x)
        return ReverseDiffTwoArgJacobianPrep(; config=config, tape=nothing)
    end
end

function DI.value_and_jacobian(
    f!, y, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    if compile
        result = jacobian!(result, prep.tape, x)
    else
        result = jacobian!(result, f!, y, x, prep.config)
    end
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian!(
    f!, y, jac, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    result = MutableDiffResult(y, (jac,))
    if compile
        result = jacobian!(result, prep.tape, x)
    else
        result = jacobian!(result, f!, y, x, prep.config)
    end
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.jacobian(
    f!, y, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        jac = jacobian!(prep.tape, x)
    else
        jac = jacobian(f!, y, x, prep.config)
    end
    return jac
end

function DI.jacobian!(
    f!, y, jac, prep::ReverseDiffTwoArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        jac = jacobian!(jac, prep.tape, x)
    else
        jac = jacobian!(jac, f!, y, x, prep.config)
    end
    return jac
end

### With contexts

function DI.prepare_jacobian(
    f!, y, ::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    config = JacobianConfig(y, x)
    return ReverseDiffTwoArgJacobianPrep(; config=config, tape=nothing)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::ReverseDiffTwoArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::ReverseDiffTwoArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    result = MutableDiffResult(y, (jac,))
    result = jacobian!(result, fc!, y, x, prep.config)
    return DiffResults.value(result), DiffResults.derivative(result)
end

function DI.jacobian(
    f!,
    y,
    prep::ReverseDiffTwoArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = jacobian(fc!, y, x, prep.config)
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::ReverseDiffTwoArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc! = DI.with_contexts(f!, contexts...)
    jac = jacobian!(jac, fc!, y, x, prep.config)
    return jac
end
