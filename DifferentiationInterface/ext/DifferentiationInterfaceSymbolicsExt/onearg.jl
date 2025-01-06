## Pushforward

struct SymbolicsOneArgPushforwardPrep{E1,E1!} <: DI.PushforwardPrep
    pf_exe::E1
    pf_exe!::E1!
end

function DI.prepare_pushforward(f, ::AutoSymbolics, x, tx::NTuple)
    dx = first(tx)
    x_var = if x isa Number
        variable(:x)
    else
        variables(:x, axes(x)...)
    end
    dx_var = if dx isa Number
        variable(:dx)
    else
        variables(:dx, axes(dx)...)
    end
    t_var = variable(:t)
    step_der_var = derivative(f(x_var + t_var * dx_var), t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    res = build_function(pf_var, vcat(myvec(x_var), myvec(dx_var)); expression=Val(false))
    (pf_exe, pf_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgPushforwardPrep(pf_exe, pf_exe!)
end

function DI.pushforward(
    f, prep::SymbolicsOneArgPushforwardPrep, ::AutoSymbolics, x, tx::NTuple
)
    ty = map(tx) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        dy = prep.pf_exe(v_vec)
    end
    return ty
end

function DI.pushforward!(
    f, ty::NTuple, prep::SymbolicsOneArgPushforwardPrep, ::AutoSymbolics, x, tx::NTuple
)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        v_vec = vcat(myvec(x), myvec(dx))
        prep.pf_exe!(dy, v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f, prep::SymbolicsOneArgPushforwardPrep, backend::AutoSymbolics, x, tx::NTuple
)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::SymbolicsOneArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
)
    return f(x), DI.pushforward!(f, ty, prep, backend, x, tx)
end

## Derivative

struct SymbolicsOneArgDerivativePrep{E1,E1!} <: DI.DerivativePrep
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(f, ::AutoSymbolics, x)
    x_var = variable(:x)
    der_var = derivative(f(x_var), x_var)

    res = build_function(der_var, x_var; expression=Val(false))
    (der_exe, der_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    return SymbolicsOneArgDerivativePrep(der_exe, der_exe!)
end

function DI.derivative(f, prep::SymbolicsOneArgDerivativePrep, ::AutoSymbolics, x)
    return prep.der_exe(x)
end

function DI.derivative!(f, der, prep::SymbolicsOneArgDerivativePrep, ::AutoSymbolics, x)
    prep.der_exe!(der, x)
    return der
end

function DI.value_and_derivative(
    f, prep::SymbolicsOneArgDerivativePrep, backend::AutoSymbolics, x
)
    return f(x), DI.derivative(f, prep, backend, x)
end

function DI.value_and_derivative!(
    f, der, prep::SymbolicsOneArgDerivativePrep, backend::AutoSymbolics, x
)
    return f(x), DI.derivative!(f, der, prep, backend, x)
end

## Gradient

struct SymbolicsOneArgGradientPrep{E1,E1!} <: DI.GradientPrep
    grad_exe::E1
    grad_exe!::E1!
end

function DI.prepare_gradient(f, ::AutoSymbolics, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.gradient only accepts vectors
    grad_var = gradient(f(x_var), vec(x_var))

    res = build_function(grad_var, vec(x_var); expression=Val(false))
    (grad_exe, grad_exe!) = res
    return SymbolicsOneArgGradientPrep(grad_exe, grad_exe!)
end

function DI.gradient(f, prep::SymbolicsOneArgGradientPrep, ::AutoSymbolics, x)
    return reshape(prep.grad_exe(vec(x)), size(x))
end

function DI.gradient!(f, grad, prep::SymbolicsOneArgGradientPrep, ::AutoSymbolics, x)
    prep.grad_exe!(vec(grad), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, prep::SymbolicsOneArgGradientPrep, backend::AutoSymbolics, x
)
    return f(x), DI.gradient(f, prep, backend, x)
end

function DI.value_and_gradient!(
    f, grad, prep::SymbolicsOneArgGradientPrep, backend::AutoSymbolics, x
)
    return f(x), DI.gradient!(f, grad, prep, backend, x)
end

## Jacobian

struct SymbolicsOneArgJacobianPrep{E1,E1!} <: DI.JacobianPrep
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f, backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}}, x
)
    x_var = variables(:x, axes(x)...)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(f(x_var)), vec(x_var))
    else
        jacobian(f(x_var), x_var)
    end

    res = build_function(jac_var, x_var; expression=Val(false))
    (jac_exe, jac_exe!) = res
    return SymbolicsOneArgJacobianPrep(jac_exe, jac_exe!)
end

function DI.jacobian(
    f,
    prep::SymbolicsOneArgJacobianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return prep.jac_exe(x)
end

function DI.jacobian!(
    f,
    jac,
    prep::SymbolicsOneArgJacobianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    prep.jac_exe!(jac, x)
    return jac
end

function DI.value_and_jacobian(
    f,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return f(x), DI.jacobian(f, prep, backend, x)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::SymbolicsOneArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return f(x), DI.jacobian!(f, jac, prep, backend, x)
end

## Hessian

struct SymbolicsOneArgHessianPrep{G,E2,E2!} <: DI.HessianPrep
    gradient_prep::G
    hess_exe::E2
    hess_exe!::E2!
end

function DI.prepare_hessian(f, backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}}, x)
    x_var = variables(:x, axes(x)...)
    # Symbolic.hessian only accepts vectors
    hess_var = if backend isa AutoSparse
        sparsehessian(f(x_var), vec(x_var))
    else
        hessian(f(x_var), vec(x_var))
    end

    res = build_function(hess_var, vec(x_var); expression=Val(false))
    (hess_exe, hess_exe!) = res

    gradient_prep = DI.prepare_gradient(f, dense_ad(backend), x)
    return SymbolicsOneArgHessianPrep(gradient_prep, hess_exe, hess_exe!)
end

function DI.hessian(
    f,
    prep::SymbolicsOneArgHessianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return prep.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    prep::SymbolicsOneArgHessianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    prep.hess_exe!(hess, vec(x))
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    y, grad = DI.value_and_gradient(f, prep.gradient_prep, dense_ad(backend), x)
    hess = DI.hessian(f, prep, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::SymbolicsOneArgHessianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    y, _ = DI.value_and_gradient!(f, grad, prep.gradient_prep, dense_ad(backend), x)
    DI.hessian!(f, hess, prep, backend, x)
    return y, grad, hess
end

## HVP

struct SymbolicsOneArgHVPPrep{G,E2,E2!} <: DI.HVPPrep
    gradient_prep::G
    hvp_exe::E2
    hvp_exe!::E2!
end

function DI.prepare_hvp(f, backend::AutoSymbolics, x, tx::NTuple)
    x_var = variables(:x, axes(x)...)
    dx_var = variables(:dx, axes(x)...)
    # Symbolic.hessian only accepts vectors
    hess_var = hessian(f(x_var), vec(x_var))
    hvp_vec_var = hess_var * vec(dx_var)

    res = build_function(hvp_vec_var, vcat(vec(x_var), vec(dx_var)); expression=Val(false))
    (hvp_exe, hvp_exe!) = res

    gradient_prep = DI.prepare_gradient(f, backend, x)
    return SymbolicsOneArgHVPPrep(gradient_prep, hvp_exe, hvp_exe!)
end

function DI.hvp(f, prep::SymbolicsOneArgHVPPrep, ::AutoSymbolics, x, tx::NTuple)
    return map(tx) do dx
        v_vec = vcat(vec(x), vec(dx))
        dg_vec = prep.hvp_exe(v_vec)
        reshape(dg_vec, size(x))
    end
end

function DI.hvp!(
    f, tg::NTuple, prep::SymbolicsOneArgHVPPrep, ::AutoSymbolics, x, tx::NTuple
)
    for b in eachindex(tx, tg)
        dx, dg = tx[b], tg[b]
        v_vec = vcat(vec(x), vec(dx))
        prep.hvp_exe!(vec(dg), v_vec)
    end
    return tg
end

function DI.gradient_and_hvp(
    f, prep::SymbolicsOneArgHVPPrep, backend::AutoSymbolics, x, tx::NTuple
)
    tg = DI.hvp(f, prep, backend, x, tx)
    grad = DI.gradient(f, prep.gradient_prep, backend, x)
    return grad, tg
end

function DI.gradient_and_hvp!(
    f, grad, tg::NTuple, prep::SymbolicsOneArgHVPPrep, backend::AutoSymbolics, x, tx::NTuple
)
    DI.hvp!(f, tg, prep, backend, x, tx)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x)
    return grad, tg
end

## Second derivative

struct SymbolicsOneArgSecondDerivativePrep{D,E1,E1!} <: DI.SecondDerivativePrep
    derivative_prep::D
    der2_exe::E1
    der2_exe!::E1!
end

function DI.prepare_second_derivative(f, backend::AutoSymbolics, x)
    x_var = variable(:x)
    der_var = derivative(f(x_var), x_var)
    der2_var = derivative(der_var, x_var)

    res = build_function(der2_var, x_var; expression=Val(false))
    (der2_exe, der2_exe!) = if res isa Tuple
        res
    elseif res isa RuntimeGeneratedFunction
        res, nothing
    end
    derivative_prep = DI.prepare_derivative(f, backend, x)
    return SymbolicsOneArgSecondDerivativePrep(derivative_prep, der2_exe, der2_exe!)
end

function DI.second_derivative(
    f, prep::SymbolicsOneArgSecondDerivativePrep, ::AutoSymbolics, x
)
    return prep.der2_exe(x)
end

function DI.second_derivative!(
    f, der2, prep::SymbolicsOneArgSecondDerivativePrep, ::AutoSymbolics, x
)
    prep.der2_exe!(der2, x)
    return der2
end

function DI.value_derivative_and_second_derivative(
    f, prep::SymbolicsOneArgSecondDerivativePrep, backend::AutoSymbolics, x
)
    y, der = DI.value_and_derivative(f, prep.derivative_prep, backend, x)
    der2 = DI.second_derivative(f, prep, backend, x)
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f, der, der2, prep::SymbolicsOneArgSecondDerivativePrep, backend::AutoSymbolics, x
)
    y, _ = DI.value_and_derivative!(f, der, prep.derivative_prep, backend, x)
    DI.second_derivative!(f, der2, prep, backend, x)
    return y, der, der2
end
