## Pullback

function DI.prepare_pullback(
    f, ::AutoReverseDiff, x, ty::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    return DI.NoPullbackPrep()
end

function DI.value_and_pullback(
    f,
    ::DI.NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    dotclosure(z, dy) = dot(fc(z), dy)
    tx = map(ty) do dy
        if y isa Number
            dy .* gradient(fc, x)
        elseif y isa AbstractArray
            gradient(Fix2(dotclosure, dy), x)
        end
    end
    return y, tx
end

function DI.value_and_pullback!(
    f,
    tx::NTuple,
    ::DI.NoPullbackPrep,
    ::AutoReverseDiff,
    x::AbstractArray,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    dotclosure(z, dy) = dot(fc(z), dy)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        if y isa Number
            dx = gradient!(dx, fc, x)
            dx .*= dy
        elseif y isa AbstractArray
            gradient!(dx, Fix2(dotclosure, dy), x)
        end
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    ::DI.NoPullbackPrep,
    backend::AutoReverseDiff,
    x::Number,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    x_array = [x]
    f_array(x_array, args...) = f(only(x_array), args...)
    y, tx_array = DI.value_and_pullback(f_array, backend, x_array, ty, contexts...)
    return y, only.(tx_array)
end

## Gradient

### Without contexts

@kwdef struct ReverseDiffGradientPrep{C,T} <: DI.GradientPrep
    config::C
    tape::T
end

function DI.prepare_gradient(f, ::AutoReverseDiff{compile}, x) where {compile}
    if compile
        tape = ReverseDiff.compile(GradientTape(f, x))
        return ReverseDiffGradientPrep(; config=nothing, tape=tape)
    else
        config = GradientConfig(x)
        return ReverseDiffGradientPrep(; config=config, tape=nothing)
    end
end

function DI.value_and_gradient!(
    f, grad, prep::ReverseDiffGradientPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    # DiffResult doesn't work because of ReverseDiff#251
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    if compile
        result = gradient!(result, prep.tape, x)
    else
        result = gradient!(result, f, x, prep.config)
    end
    return DR.value(result), DR.gradient(result)
end

function DI.value_and_gradient(
    f, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff{compile}, x
) where {compile}
    # GradientResult doesn't work because it tries to mutate an SArray
    result = MutableDiffResult(zero(eltype(x)), (similar(x),))
    if compile
        result = gradient!(result, prep.tape, x)
    else
        result = gradient!(result, f, x, prep.config)
    end
    return DR.value(result), DR.gradient(result)
end

function DI.gradient!(
    f, grad, prep::ReverseDiffGradientPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return gradient!(grad, prep.tape, x)
    else
        return gradient!(grad, f, x, prep.config)
    end
end

function DI.gradient(
    f, prep::ReverseDiffGradientPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return gradient!(prep.tape, x)
    else
        return gradient(f, x, prep.config)
    end
end

### With contexts

function DI.prepare_gradient(
    f, ::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    config = GradientConfig(x)
    return ReverseDiffGradientPrep(; config=config, tape=nothing)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::ReverseDiffGradientPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    # DiffResult doesn't work because of ReverseDiff#251
    result = MutableDiffResult(zero(eltype(x)), (grad,))
    result = gradient!(result, fc, x, prep.config)
    return DR.value(result), DR.gradient(result)
end

function DI.value_and_gradient(
    f, prep::ReverseDiffGradientPrep, ::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    fc = DI.with_contexts(f, contexts...)
    # GradientResult doesn't work because it tries to mutate an SArray
    result = MutableDiffResult(zero(eltype(x)), (similar(x),))
    result = gradient!(result, fc, x, prep.config)
    return DR.value(result), DR.gradient(result)
end

function DI.gradient!(
    f,
    grad,
    prep::ReverseDiffGradientPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return gradient!(grad, fc, x, prep.config)
end

function DI.gradient(
    f, prep::ReverseDiffGradientPrep, ::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return gradient(fc, x, prep.config)
end

## Jacobian

### Without contexts

@kwdef struct ReverseDiffOneArgJacobianPrep{C,T} <: DI.JacobianPrep
    config::C
    tape::T
end

function DI.prepare_jacobian(f, ::AutoReverseDiff{compile}, x) where {compile}
    if compile
        tape = ReverseDiff.compile(JacobianTape(f, x))
        return ReverseDiffOneArgJacobianPrep(; config=nothing, tape=tape)
    else
        config = JacobianConfig(x)
        return ReverseDiffOneArgJacobianPrep(; config=config, tape=nothing)
    end
end

function DI.value_and_jacobian!(
    f, jac, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    y = f(x)
    result = DiffResult(y, (jac,))
    if compile
        result = jacobian!(result, prep.tape, x)
    else
        result = jacobian!(result, f, x, prep.config)
    end
    y = DR.value(result)
    jac === DR.jacobian(result) || copyto!(jac, DR.jacobian(result))
    return y, jac
end

function DI.value_and_jacobian(
    f, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return f(x), jacobian!(prep.tape, x)
    else
        return f(x), jacobian(f, x, prep.config)
    end
end

function DI.jacobian!(
    f, jac, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return jacobian!(jac, prep.tape, x)
    else
        return jacobian!(jac, f, x, prep.config)
    end
end

function DI.jacobian(
    f, prep::ReverseDiffOneArgJacobianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return jacobian!(prep.tape, x)
    else
        return jacobian(f, x, prep.config)
    end
end

### With contexts

function DI.prepare_jacobian(
    f, ::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    config = JacobianConfig(x)
    return ReverseDiffOneArgJacobianPrep(; config=config, tape=nothing)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::ReverseDiffOneArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    y = fc(x)
    result = DiffResult(y, (jac,))
    result = jacobian!(result, fc, x, prep.config)
    y = DR.value(result)
    jac === DR.jacobian(result) || copyto!(jac, DR.jacobian(result))
    return y, jac
end

function DI.value_and_jacobian(
    f,
    prep::ReverseDiffOneArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return fc(x), jacobian(fc, x, prep.config)
end

function DI.jacobian!(
    f,
    jac,
    prep::ReverseDiffOneArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return jacobian!(jac, fc, x, prep.config)
end

function DI.jacobian(
    f,
    prep::ReverseDiffOneArgJacobianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return jacobian(fc, x, prep.config)
end

## Hessian

### Without contexts

#=
At the moment, all three components of `value_gradient_and_hessian` are computed separately. In theory, `hessian!(result::DiffResult, ...)` is supposed to manage everything at once, but in practice I often find that the value itself is not updated, even when I call `DI.value_and_gradient` separately.
=#

@kwdef struct ReverseDiffHessianPrep{G<:ReverseDiffGradientPrep,HC,HT} <: DI.HessianPrep
    gradient_prep::G
    hessian_config::HC
    hessian_tape::HT
end

function DI.prepare_hessian(f, backend::AutoReverseDiff{compile}, x) where {compile}
    gradient_prep = DI.prepare_gradient(f, backend, x)
    if compile
        hessian_tape = ReverseDiff.compile(HessianTape(f, x))
        return ReverseDiffHessianPrep(;
            gradient_prep, hessian_config=nothing, hessian_tape=hessian_tape
        )
    else
        hessian_config = HessianConfig(x)
        return ReverseDiffHessianPrep(;
            gradient_prep, hessian_config=hessian_config, hessian_tape=nothing
        )
    end
end

function DI.hessian!(
    f, hess, prep::ReverseDiffHessianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return hessian!(hess, prep.hessian_tape, x)
    else
        return hessian!(hess, f, x, prep.hessian_config)
    end
end

function DI.hessian(
    f, prep::ReverseDiffHessianPrep, ::AutoReverseDiff{compile}, x
) where {compile}
    if compile
        return hessian!(prep.hessian_tape, x)
    else
        return hessian(f, x, prep.hessian_config)
    end
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, prep::ReverseDiffHessianPrep, backend::AutoReverseDiff{compile}, x
) where {compile}
    y = f(x)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x)
    DI.hessian!(f, hess, prep, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian(
    f, prep::ReverseDiffHessianPrep, backend::AutoReverseDiff{compile}, x
) where {compile}
    y = f(x)
    grad = DI.gradient(f, prep.gradient_prep, backend, x)
    hess = DI.hessian(f, prep, backend, x)
    return y, grad, hess
end

### With contexts

function DI.prepare_hessian(
    f, backend::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    gradient_prep = DI.prepare_gradient(f, backend, x, contexts...)
    hessian_config = HessianConfig(x)
    return ReverseDiffHessianPrep(;
        gradient_prep, hessian_config=hessian_config, hessian_tape=nothing
    )
end

function DI.hessian!(
    f,
    hess,
    prep::ReverseDiffHessianPrep,
    ::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return hessian!(hess, fc, x, prep.hessian_config)
end

function DI.hessian(
    f, prep::ReverseDiffHessianPrep, ::AutoReverseDiff, x, contexts::Vararg{DI.Context,C}
) where {C}
    fc = DI.with_contexts(f, contexts...)
    return hessian(fc, x, prep.hessian_config)
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::ReverseDiffHessianPrep,
    backend::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    y = f(x, map(DI.unwrap, contexts)...)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x, contexts...)
    DI.hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian(
    f,
    prep::ReverseDiffHessianPrep,
    backend::AutoReverseDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    y = f(x, map(DI.unwrap, contexts)...)
    grad = DI.gradient(f, prep.gradient_prep, backend, x, contexts...)
    hess = DI.hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end
