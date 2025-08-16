## Pushforward

struct FastDifferentiationOneArgPushforwardPrep{SIG,Y,E1,E1!} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    y_prototype::Y
    jvp_exe::E1
    jvp_exe!::E1!
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    y_prototype = f(x, map(DI.unwrap, contexts)...)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)
    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(
        jv_vec_var, x_vec_var, v_vec_var, context_vec_vars...; in_place=false
    )
    jvp_exe! = make_function(
        jv_vec_var, x_vec_var, v_vec_var, context_vec_vars...; in_place=true
    )
    return FastDifferentiationOneArgPushforwardPrep(_sig, y_prototype, jvp_exe, jvp_exe!)
end

function DI.pushforward(
    f,
    prep::FastDifferentiationOneArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ty = map(tx) do dx
        result = prep.jvp_exe(myvec(x), myvec(dx), map(myvec_unwrap, contexts)...)
        if prep.y_prototype isa Number
            return only(result)
        else
            return reshape(result, size(prep.y_prototype))
        end
    end
    return ty
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::FastDifferentiationOneArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        prep.jvp_exe!(myvec(dy), myvec(x), myvec(dx), map(myvec_unwrap, contexts)...)
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::FastDifferentiationOneArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pushforward(f, prep, backend, x, tx, contexts...)
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::FastDifferentiationOneArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pushforward!(f, ty, prep, backend, x, tx, contexts...)
end

## Pullback

struct FastDifferentiationOneArgPullbackPrep{SIG,E1,E1!} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    vjp_exe::E1
    vjp_exe!::E1!
end

function DI.prepare_pullback_nokwarg(
    strict::Val,
    f,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    vj_vec_var, v_vec_var = jacobian_transpose_v(y_vec_var, x_vec_var)
    vjp_exe = make_function(
        vj_vec_var, x_vec_var, v_vec_var, context_vec_vars...; in_place=false
    )
    vjp_exe! = make_function(
        vj_vec_var, x_vec_var, v_vec_var, context_vec_vars...; in_place=true
    )
    return FastDifferentiationOneArgPullbackPrep(_sig, vjp_exe, vjp_exe!)
end

function DI.pullback(
    f,
    prep::FastDifferentiationOneArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    tx = map(ty) do dy
        result = prep.vjp_exe(myvec(x), myvec(dy), map(myvec_unwrap, contexts)...)
        if x isa Number
            return only(result)
        else
            return reshape(result, size(x))
        end
    end
    return tx
end

function DI.pullback!(
    f,
    tx::NTuple,
    prep::FastDifferentiationOneArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        prep.vjp_exe!(myvec(dx), myvec(x), myvec(dy), map(myvec_unwrap, contexts)...)
    end
    return tx
end

function DI.value_and_pullback(
    f,
    prep::FastDifferentiationOneArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pullback(f, prep, backend, x, ty, contexts...)
end

function DI.value_and_pullback!(
    f,
    tx::NTuple,
    prep::FastDifferentiationOneArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pullback!(f, tx, prep, backend, x, ty, contexts...)
end

## Derivative

struct FastDifferentiationOneArgDerivativePrep{SIG,Y,E1,E1!} <: DI.DerivativePrep{SIG}
    _sig::Val{SIG}
    y_prototype::Y
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative_nokwarg(
    strict::Val, f, backend::AutoFastDifferentiation, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    y_prototype = f(x, map(DI.unwrap, contexts)...)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var, context_vec_vars...; in_place=false)
    der_exe! = make_function(der_vec_var, x_vec_var, context_vec_vars...; in_place=true)
    return FastDifferentiationOneArgDerivativePrep(_sig, y_prototype, der_exe, der_exe!)
end

function DI.derivative(
    f,
    prep::FastDifferentiationOneArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    result = prep.der_exe(myvec(x), map(myvec_unwrap, contexts)...)
    if prep.y_prototype isa Number
        return only(result)
    else
        return reshape(result, size(prep.y_prototype))
    end
end

function DI.derivative!(
    f,
    der,
    prep::FastDifferentiationOneArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.der_exe!(myvec(der), myvec(x), map(myvec_unwrap, contexts)...)
    return der
end

function DI.value_and_derivative(
    f,
    prep::FastDifferentiationOneArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.derivative(f, prep, backend, x, contexts...)
end

function DI.value_and_derivative!(
    f,
    der,
    prep::FastDifferentiationOneArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.derivative!(f, der, prep, backend, x, contexts...)
end

## Gradient

struct FastDifferentiationOneArgGradientPrep{SIG,E1,E1!} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_gradient_nokwarg(
    strict::Val, f, backend::AutoFastDifferentiation, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    jac_var = jacobian(y_vec_var, x_vec_var)
    jac_exe = make_function(jac_var, x_vec_var, context_vec_vars...; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var, context_vec_vars...; in_place=true)
    return FastDifferentiationOneArgGradientPrep(_sig, jac_exe, jac_exe!)
end

function DI.gradient(
    f,
    prep::FastDifferentiationOneArgGradientPrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    jac = prep.jac_exe(myvec(x), map(myvec_unwrap, contexts)...)
    grad_vec = @view jac[1, :]
    return reshape(grad_vec, size(x))
end

function DI.gradient!(
    f,
    grad,
    prep::FastDifferentiationOneArgGradientPrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.jac_exe!(reshape(grad, 1, length(grad)), myvec(x), map(myvec_unwrap, contexts)...)
    return grad
end

function DI.value_and_gradient(
    f,
    prep::FastDifferentiationOneArgGradientPrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...), DI.gradient(f, prep, backend, x, contexts...)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::FastDifferentiationOneArgGradientPrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.gradient!(f, grad, prep, backend, x, contexts...)
end

## Jacobian

struct FastDifferentiationOneArgJacobianPrep{SIG,Y,P,E1,E1!} <: DI.SparseJacobianPrep{SIG}
    _sig::Val{SIG}
    y_prototype::Y
    sparsity::P
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian_nokwarg(
    strict::Val,
    f,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    y_prototype = f(x, map(DI.unwrap, contexts)...)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    if backend isa AutoSparse
        jac_var = sparse_jacobian(y_vec_var, x_vec_var)
        sparsity = DI.get_pattern(jac_var)
    else
        jac_var = jacobian(y_vec_var, x_vec_var)
        sparsity = nothing
    end
    jac_exe = make_function(jac_var, x_vec_var, context_vec_vars...; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var, context_vec_vars...; in_place=true)
    return FastDifferentiationOneArgJacobianPrep(
        _sig, y_prototype, sparsity, jac_exe, jac_exe!
    )
end

function DI.jacobian(
    f,
    prep::FastDifferentiationOneArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return prep.jac_exe(myvec(x), map(myvec_unwrap, contexts)...)
end

function DI.jacobian!(
    f,
    jac,
    prep::FastDifferentiationOneArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.jac_exe!(jac, myvec(x), map(myvec_unwrap, contexts)...)
    return jac
end

function DI.value_and_jacobian(
    f,
    prep::FastDifferentiationOneArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...), DI.jacobian(f, prep, backend, x, contexts...)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::FastDifferentiationOneArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.jacobian!(f, jac, prep, backend, x, contexts...)
end

## Second derivative

struct FastDifferentiationAllocatingSecondDerivativePrep{SIG,Y,D,E2,E2!} <:
       DI.SecondDerivativePrep{SIG}
    _sig::Val{SIG}
    y_prototype::Y
    derivative_prep::D
    der2_exe::E2
    der2_exe!::E2!
end

function DI.prepare_second_derivative_nokwarg(
    strict::Val, f, backend::AutoFastDifferentiation, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    y_prototype = f(x, map(DI.unwrap, contexts)...)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)

    der2_vec_var = derivative(y_vec_var, x_var, x_var)
    der2_exe = make_function(der2_vec_var, x_vec_var, context_vec_vars...; in_place=false)
    der2_exe! = make_function(der2_vec_var, x_vec_var, context_vec_vars...; in_place=true)

    derivative_prep = DI.prepare_derivative_nokwarg(strict, f, backend, x, contexts...)
    return FastDifferentiationAllocatingSecondDerivativePrep(
        _sig, y_prototype, derivative_prep, der2_exe, der2_exe!
    )
end

function DI.second_derivative(
    f,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    result = prep.der2_exe(myvec(x), map(myvec_unwrap, contexts)...)
    if prep.y_prototype isa Number
        return only(result)
    else
        return reshape(result, size(prep.y_prototype))
    end
end

function DI.second_derivative!(
    f,
    der2,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.der2_exe!(myvec(der2), myvec(x), map(myvec_unwrap, contexts)...)
    return der2
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, der = DI.value_and_derivative(f, prep.derivative_prep, backend, x, contexts...)
    der2 = DI.second_derivative(f, prep, backend, x, contexts...)
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, _ = DI.value_and_derivative!(f, der, prep.derivative_prep, backend, x, contexts...)
    DI.second_derivative!(f, der2, prep, backend, x, contexts...)
    return y, der, der2
end

## HVP

struct FastDifferentiationHVPPrep{SIG,E2,E2!,E1} <: DI.HVPPrep{SIG}
    sig::Val{SIG}
    hvp_exe::E2
    hvp_exe!::E2!
    gradient_prep::E1
end

function DI.prepare_hvp_nokwarg(
    strict::Val,
    f,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    hv_vec_var, v_vec_var = hessian_times_v(y_var, x_vec_var)
    hvp_exe = make_function(
        hv_vec_var, x_vec_var, v_vec_var, context_vec_vars...; in_place=false
    )
    hvp_exe! = make_function(
        hv_vec_var, x_vec_var, v_vec_var, context_vec_vars...; in_place=true
    )

    gradient_prep = DI.prepare_gradient_nokwarg(strict, f, backend, x, contexts...)
    return FastDifferentiationHVPPrep(_sig, hvp_exe, hvp_exe!, gradient_prep)
end

function DI.hvp(
    f,
    prep::FastDifferentiationHVPPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    tg = map(tx) do dx
        dg_vec = prep.hvp_exe(myvec(x), myvec(dx), map(myvec_unwrap, contexts)...)
        return reshape(dg_vec, size(x))
    end
    return tg
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::FastDifferentiationHVPPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    for b in eachindex(tx, tg)
        dx, dg = tx[b], tg[b]
        prep.hvp_exe!(myvec(dg), myvec(x), myvec(dx), map(myvec_unwrap, contexts)...)
    end
    return tg
end

function DI.gradient_and_hvp(
    f,
    prep::FastDifferentiationHVPPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    tg = DI.hvp(f, prep, backend, x, tx, contexts...)
    grad = DI.gradient(f, prep.gradient_prep, backend, x, contexts...)
    return grad, tg
end

function DI.gradient_and_hvp!(
    f,
    grad,
    tg::NTuple,
    prep::FastDifferentiationHVPPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    DI.hvp!(f, tg, prep, backend, x, tx, contexts...)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x, contexts...)
    return grad, tg
end

## Hessian

struct FastDifferentiationHessianPrep{SIG,G,P,E2,E2!} <: DI.SparseHessianPrep{SIG}
    _sig::Val{SIG}
    gradient_prep::G
    sparsity::P
    hess_exe::E2
    hess_exe!::E2!
end

function DI.prepare_hessian_nokwarg(
    strict::Val,
    f,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = f(x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)

    if backend isa AutoSparse
        hess_var = sparse_hessian(y_var, x_vec_var)
        sparsity = DI.get_pattern(hess_var)
    else
        hess_var = hessian(y_var, x_vec_var)
        sparsity = nothing
    end
    hess_exe = make_function(hess_var, x_vec_var, context_vec_vars...; in_place=false)
    hess_exe! = make_function(hess_var, x_vec_var, context_vec_vars...; in_place=true)

    gradient_prep = DI.prepare_gradient_nokwarg(
        strict, f, dense_ad(backend), x, contexts...
    )
    return FastDifferentiationHessianPrep(
        _sig, gradient_prep, sparsity, hess_exe, hess_exe!
    )
end

function DI.hessian(
    f,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return prep.hess_exe(myvec(x), map(myvec_unwrap, contexts)...)
end

function DI.hessian!(
    f,
    hess,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.hess_exe!(hess, myvec(x), map(myvec_unwrap, contexts)...)
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, grad = DI.value_and_gradient(
        f, prep.gradient_prep, dense_ad(backend), x, contexts...
    )
    hess = DI.hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, _ = DI.value_and_gradient!(
        f, grad, prep.gradient_prep, dense_ad(backend), x, contexts...
    )
    DI.hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end
