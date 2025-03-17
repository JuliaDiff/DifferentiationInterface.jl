## Pushforward

struct SymbolicsOneArgPushforwardPrep{SIG,E1,E1!} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    pf_exe::E1
    pf_exe!::E1!
end

function DI.prepare_pushforward(
    f,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    dx = first(tx)
    x_var = variablize(x, :x)
    dx_var = variablize(dx, :dx)
    t_var = variable(:t)
    context_vars = variablize(contexts)
    step_der_var = derivative(f(x_var + t_var * dx_var, context_vars...), t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    res = build_function(pf_var, x_var, dx_var, context_vars...; expression=Val(false))
    (pf_exe, pf_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgPushforwardPrep(_sig, pf_exe, pf_exe!)
end

function DI.pushforward(
    f,
    prep::SymbolicsOneArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ty = map(tx) do dx
        dy = prep.pf_exe(x, dx, map(DI.unwrap, contexts)...)
    end
    return ty
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::SymbolicsOneArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        prep.pf_exe!(dy, x, dx, map(DI.unwrap, contexts)...)
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::SymbolicsOneArgPushforwardPrep,
    backend::AutoSymbolics,
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
    prep::SymbolicsOneArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.pushforward!(f, ty, prep, backend, x, tx, contexts...)
end

## Derivative

struct SymbolicsOneArgDerivativePrep{SIG,E1,E1!} <: DI.DerivativePrep{SIG}
    _sig::Val{SIG}
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(
    f, backend::AutoSymbolics, x, contexts::Vararg{DI.Context,C}; strict::Val=Val(false)
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    der_var = derivative(f(x_var, context_vars...), x_var)

    res = build_function(der_var, x_var, context_vars...; expression=Val(false))
    (der_exe, der_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgDerivativePrep(_sig, der_exe, der_exe!)
end

function DI.derivative(
    f,
    prep::SymbolicsOneArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return prep.der_exe(x, map(DI.unwrap, contexts)...)
end

function DI.derivative!(
    f,
    der,
    prep::SymbolicsOneArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.der_exe!(der, x, map(DI.unwrap, contexts)...)
    return der
end

function DI.value_and_derivative(
    f,
    prep::SymbolicsOneArgDerivativePrep,
    backend::AutoSymbolics,
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
    prep::SymbolicsOneArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.derivative!(f, der, prep, backend, x, contexts...)
end

## Gradient

struct SymbolicsOneArgGradientPrep{SIG,E1,E1!} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    grad_exe::E1
    grad_exe!::E1!
end

function DI.prepare_gradient(
    f, backend::AutoSymbolics, x, contexts::Vararg{DI.Context,C}; strict::Val=Val(false)
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    # Symbolic.gradient only accepts vectors
    grad_var = gradient(f(x_var, context_vars...), vec(x_var))

    res = build_function(grad_var, vec(x_var), context_vars...; expression=Val(false))
    (grad_exe, grad_exe!) = res
    return SymbolicsOneArgGradientPrep(_sig, grad_exe, grad_exe!)
end

function DI.gradient(
    f,
    prep::SymbolicsOneArgGradientPrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return reshape(prep.grad_exe(vec(x), map(DI.unwrap, contexts)...), size(x))
end

function DI.gradient!(
    f,
    grad,
    prep::SymbolicsOneArgGradientPrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.grad_exe!(vec(grad), vec(x), map(DI.unwrap, contexts)...)
    return grad
end

function DI.value_and_gradient(
    f,
    prep::SymbolicsOneArgGradientPrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...), DI.gradient(f, prep, backend, x, contexts...)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::SymbolicsOneArgGradientPrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.gradient!(f, grad, prep, backend, x, contexts...)
end

## Jacobian

struct SymbolicsOneArgJacobianPrep{SIG,E1,E1!} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(f(x_var, context_vars...)), vec(x_var))
    else
        jacobian(f(x_var, context_vars...), x_var)
    end

    res = build_function(jac_var, x_var, context_vars...; expression=Val(false))
    (jac_exe, jac_exe!) = res
    return SymbolicsOneArgJacobianPrep(_sig, jac_exe, jac_exe!)
end

function DI.jacobian(
    f,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, backend, x, contexts...)
    return prep.jac_exe(x, map(DI.unwrap, contexts)...)
end

function DI.jacobian!(
    f,
    jac,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, backend, x, contexts...)
    prep.jac_exe!(jac, x, map(DI.unwrap, contexts)...)
    return jac
end

function DI.value_and_jacobian(
    f,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...), DI.jacobian(f, prep, backend, x, contexts...)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, backend, x, contexts...)
    return f(x, map(DI.unwrap, contexts)...),
    DI.jacobian!(f, jac, prep, backend, x, contexts...)
end

## Hessian

struct SymbolicsOneArgHessianPrep{SIG,G,E2,E2!} <: DI.HessianPrep{SIG}
    _sig::Val{SIG}
    gradient_prep::G
    hess_exe::E2
    hess_exe!::E2!
end

function DI.prepare_hessian(
    f,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    # Symbolic.hessian only accepts vectors
    hess_var = if backend isa AutoSparse
        sparsehessian(f(x_var, context_vars...), vec(x_var))
    else
        hessian(f(x_var, context_vars...), vec(x_var))
    end

    res = build_function(hess_var, vec(x_var), context_vars...; expression=Val(false))
    (hess_exe, hess_exe!) = res

    gradient_prep = DI.prepare_gradient(f, dense_ad(backend), x, contexts...)
    return SymbolicsOneArgHessianPrep(_sig, gradient_prep, hess_exe, hess_exe!)
end

function DI.hessian(
    f,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return prep.hess_exe(vec(x), map(DI.unwrap, contexts)...)
end

function DI.hessian!(
    f,
    hess,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.hess_exe!(hess, vec(x), map(DI.unwrap, contexts)...)
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
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
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
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

## HVP

struct SymbolicsOneArgHVPPrep{SIG,G,E2,E2!} <: DI.HVPPrep{SIG}
    _sig::Val{SIG}
    gradient_prep::G
    hvp_exe::E2
    hvp_exe!::E2!
end

function DI.prepare_hvp(
    f,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    dx = first(tx)
    x_var = variablize(x, :x)
    dx_var = variablize(dx, :dx)
    context_vars = variablize(contexts)
    # Symbolic.hessian only accepts vectors
    hess_var = hessian(f(x_var, context_vars...), vec(x_var))
    hvp_vec_var = hess_var * vec(dx_var)

    res = build_function(
        hvp_vec_var, vec(x_var), vec(dx_var), context_vars...; expression=Val(false)
    )
    (hvp_exe, hvp_exe!) = res

    gradient_prep = DI.prepare_gradient(f, backend, x, contexts...)
    return SymbolicsOneArgHVPPrep(_sig, gradient_prep, hvp_exe, hvp_exe!)
end

function DI.hvp(
    f,
    prep::SymbolicsOneArgHVPPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return map(tx) do dx
        dg_vec = prep.hvp_exe(vec(x), vec(dx), map(DI.unwrap, contexts)...)
        reshape(dg_vec, size(x))
    end
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::SymbolicsOneArgHVPPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    for b in eachindex(tx, tg)
        dx, dg = tx[b], tg[b]
        prep.hvp_exe!(vec(dg), vec(x), vec(dx), map(DI.unwrap, contexts)...)
    end
    return tg
end

function DI.gradient_and_hvp(
    f,
    prep::SymbolicsOneArgHVPPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    tg = DI.hvp(f, prep, backend, x, tx, contexts...)
    grad = DI.gradient(f, prep.gradient_prep, backend, x, contexts...)
    return grad, tg
end

function DI.gradient_and_hvp!(
    f,
    grad,
    tg::NTuple,
    prep::SymbolicsOneArgHVPPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    DI.hvp!(f, tg, prep, backend, x, tx, contexts...)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x, contexts...)
    return grad, tg
end

## Second derivative

struct SymbolicsOneArgSecondDerivativePrep{SIG,D,E1,E1!} <: DI.SecondDerivativePrep{SIG}
    _sig::Val{SIG}
    derivative_prep::D
    der2_exe::E1
    der2_exe!::E1!
end

function DI.prepare_second_derivative(
    f, backend::AutoSymbolics, x, contexts::Vararg{DI.Context,C}; strict::Val=Val(false)
) where {C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    der_var = derivative(f(x_var, context_vars...), x_var)
    der2_var = derivative(der_var, x_var)

    res = build_function(der2_var, x_var, context_vars...; expression=Val(false))
    (der2_exe, der2_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    derivative_prep = DI.prepare_derivative(f, backend, x, contexts...)
    return SymbolicsOneArgSecondDerivativePrep(_sig, derivative_prep, der2_exe, der2_exe!)
end

function DI.second_derivative(
    f,
    prep::SymbolicsOneArgSecondDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return prep.der2_exe(x, map(DI.unwrap, contexts)...)
end

function DI.second_derivative!(
    f,
    der2,
    prep::SymbolicsOneArgSecondDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    prep.der2_exe!(der2, x, map(DI.unwrap, contexts)...)
    return der2
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::SymbolicsOneArgSecondDerivativePrep,
    backend::AutoSymbolics,
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
    prep::SymbolicsOneArgSecondDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, _ = DI.value_and_derivative!(f, der, prep.derivative_prep, backend, x, contexts...)
    DI.second_derivative!(f, der2, prep, backend, x, contexts...)
    return y, der, der2
end
