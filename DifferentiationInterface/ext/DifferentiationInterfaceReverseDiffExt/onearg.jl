## Pullback

function DI.prepare_pullback_nokwarg(
        strict::Val, f, backend::AutoReverseDiff, x, ty::NTuple, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    return DI.NoPullbackPrep(_sig)
end

function DI.value_and_pullback(
        f,
        prep::DI.NoPullbackPrep,
        backend::AutoReverseDiff,
        x::AbstractArray,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
        prep::DI.NoPullbackPrep,
        backend::AutoReverseDiff,
        x::AbstractArray,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
        prep::DI.NoPullbackPrep,
        backend::AutoReverseDiff,
        x::Number,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    x_array = [x]
    f_array(x_array, args...) = f(only(x_array), args...)
    y, tx_array = DI.value_and_pullback(f_array, backend, x_array, ty, contexts...)
    return y, only.(tx_array)
end

## Gradient

### Without contexts

struct ReverseDiffGradientPrep{SIG, C, T} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    config::C
    tape::T
end

function DI.prepare_gradient_nokwarg(
        strict::Val, f, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    _sig = DI.signature(f, backend, x; strict)
    if compile
        tape = ReverseDiff.compile(GradientTape(f, x))
        return ReverseDiffGradientPrep(_sig, nothing, tape)
    else
        config = GradientConfig(x)
        return ReverseDiffGradientPrep(_sig, config, nothing)
    end
end

function DI.value_and_gradient!(
        f, grad, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    result = MutableDiffResult(zero(eltype(x)), (grad,))  # ReverseDiff#251
    if compile
        result = gradient!(result, prep.tape, x)
    else
        result = gradient!(result, f, x, prep.config)
    end
    return DR.value(result), grad  # ReverseDiff#269
end

function DI.value_and_gradient(
        f, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    # GradientResult tries to mutate an SArray
    result = MutableDiffResult(zero(eltype(x)), (similar(x),))
    if compile
        result = gradient!(result, prep.tape, x)
    else
        result = gradient!(result, f, x, prep.config)
    end
    return DR.value(result), DR.gradient(result)
end

function DI.gradient!(
        f, grad, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return gradient!(grad, prep.tape, x)
    else
        return gradient!(grad, f, x, prep.config)
    end
end

function DI.gradient(
        f, prep::ReverseDiffGradientPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return gradient!(prep.tape, x)
    else
        return gradient(f, x, prep.config)
    end
end

### With contexts

function DI.prepare_gradient_nokwarg(
        strict::Val, f, backend::AutoReverseDiff, x, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    config = GradientConfig(x)
    return ReverseDiffGradientPrep(_sig, config, nothing)
end

function DI.value_and_gradient!(
        f,
        grad,
        prep::ReverseDiffGradientPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    result = MutableDiffResult(zero(eltype(x)), (grad,))  # ReverseDiff#251
    result = gradient!(result, fc, x, prep.config)
    return DR.value(result), grad  # ReverseDiff#269
end

function DI.value_and_gradient(
        f,
        prep::ReverseDiffGradientPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    # GradientResult tries to mutate an SArray
    result = MutableDiffResult(zero(eltype(x)), (similar(x),))
    result = gradient!(result, fc, x, prep.config)
    return DR.value(result), DR.gradient(result)
end

function DI.gradient!(
        f,
        grad,
        prep::ReverseDiffGradientPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return gradient!(grad, fc, x, prep.config)
end

function DI.gradient(
        f,
        prep::ReverseDiffGradientPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return gradient(fc, x, prep.config)
end

## Jacobian

### Without contexts

struct ReverseDiffOneArgJacobianPrep{SIG, C, T} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    config::C
    tape::T
end

function DI.prepare_jacobian_nokwarg(
        strict::Val, f, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    _sig = DI.signature(f, backend, x; strict)
    if compile
        tape = ReverseDiff.compile(JacobianTape(f, x))
        return ReverseDiffOneArgJacobianPrep(_sig, nothing, tape)
    else
        config = JacobianConfig(x)
        return ReverseDiffOneArgJacobianPrep(_sig, config, nothing)
    end
end

function DI.value_and_jacobian!(
        f, jac, prep::ReverseDiffOneArgJacobianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
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
        f, prep::ReverseDiffOneArgJacobianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return f(x), jacobian!(prep.tape, x)
    else
        return f(x), jacobian(f, x, prep.config)
    end
end

function DI.jacobian!(
        f, jac, prep::ReverseDiffOneArgJacobianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return jacobian!(jac, prep.tape, x)
    else
        return jacobian!(jac, f, x, prep.config)
    end
end

function DI.jacobian(
        f, prep::ReverseDiffOneArgJacobianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return jacobian!(prep.tape, x)
    else
        return jacobian(f, x, prep.config)
    end
end

### With contexts

function DI.prepare_jacobian_nokwarg(
        strict::Val, f, backend::AutoReverseDiff, x, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    config = JacobianConfig(x)
    return ReverseDiffOneArgJacobianPrep(_sig, config, nothing)
end

function DI.value_and_jacobian!(
        f,
        jac,
        prep::ReverseDiffOneArgJacobianPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
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
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return fc(x), jacobian(fc, x, prep.config)
end

function DI.jacobian!(
        f,
        jac,
        prep::ReverseDiffOneArgJacobianPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return jacobian!(jac, fc, x, prep.config)
end

function DI.jacobian(
        f,
        prep::ReverseDiffOneArgJacobianPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return jacobian(fc, x, prep.config)
end

## Hessian

### Without contexts

struct ReverseDiffHessianPrep{SIG, G <: ReverseDiffGradientPrep, HC, HT} <: DI.HessianPrep{SIG}
    _sig::Val{SIG}
    gradient_prep::G
    hessian_config::HC
    hessian_tape::HT
end

function DI.prepare_hessian_nokwarg(
        strict::Val, f, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    _sig = DI.signature(f, backend, x; strict)
    gradient_prep = DI.prepare_gradient_nokwarg(strict, f, backend, x)
    if compile
        hessian_tape = ReverseDiff.compile(HessianTape(f, x))
        return ReverseDiffHessianPrep(_sig, gradient_prep, nothing, hessian_tape)
    else
        hessian_config = HessianConfig(x)
        return ReverseDiffHessianPrep(_sig, gradient_prep, hessian_config, nothing)
    end
end

function DI.hessian!(
        f, hess, prep::ReverseDiffHessianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return hessian!(hess, prep.hessian_tape, x)
    else
        return hessian!(hess, f, x, prep.hessian_config)
    end
end

function DI.hessian(
        f, prep::ReverseDiffHessianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    if compile
        return hessian!(prep.hessian_tape, x)
    else
        return hessian(f, x, prep.hessian_config)
    end
end

function DI.value_gradient_and_hessian!(
        f, grad, hess, prep::ReverseDiffHessianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    y = f(x)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x)
    DI.hessian!(f, hess, prep, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian(
        f, prep::ReverseDiffHessianPrep, backend::AutoReverseDiff{compile}, x
    ) where {compile}
    DI.check_prep(f, prep, backend, x)
    y = f(x)
    grad = DI.gradient(f, prep.gradient_prep, backend, x)
    hess = DI.hessian(f, prep, backend, x)
    return y, grad, hess
end

### With contexts

function DI.prepare_hessian_nokwarg(
        strict::Val, f, backend::AutoReverseDiff, x, contexts::Vararg{DI.Context, C}
    ) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    gradient_prep = DI.prepare_gradient_nokwarg(strict, f, backend, x, contexts...)
    hessian_config = HessianConfig(x)
    return ReverseDiffHessianPrep(_sig, gradient_prep, hessian_config, nothing)
end

function DI.hessian!(
        f,
        hess,
        prep::ReverseDiffHessianPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return hessian!(hess, fc, x, prep.hessian_config)
end

function DI.hessian(
        f,
        prep::ReverseDiffHessianPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = DI.fix_tail(f, map(DI.unwrap, contexts)...)
    return hessian(fc, x, prep.hessian_config)
end

function DI.value_gradient_and_hessian!(
        f,
        grad,
        hess,
        prep::ReverseDiffHessianPrep,
        backend::AutoReverseDiff,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
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
        contexts::Vararg{DI.Context, C},
    ) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y = f(x, map(DI.unwrap, contexts)...)
    grad = DI.gradient(f, prep.gradient_prep, backend, x, contexts...)
    hess = DI.hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end
