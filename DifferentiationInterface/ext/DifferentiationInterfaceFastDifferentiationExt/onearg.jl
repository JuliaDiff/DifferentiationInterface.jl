## Pushforward

struct FastDifferentiationOneArgPushforwardPrep{Y,E1,E1!} <: DI.PushforwardPrep
    y_prototype::Y
    jvp_exe::E1
    jvp_exe!::E1!
end

function DI.prepare_pushforward(f, ::AutoFastDifferentiation, x, tx::NTuple)
    y_prototype = f(x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    jvp_exe! = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationOneArgPushforwardPrep(y_prototype, jvp_exe, jvp_exe!)
end

function DI.pushforward(
    f,
    prep::FastDifferentiationOneArgPushforwardPrep,
    ::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    ty = map(tx) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        if prep.y_prototype isa Number
            return only(prep.jvp_exe(v_vec))
        else
            return reshape(prep.jvp_exe(v_vec), size(prep.y_prototype))
        end
    end
    return ty
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::FastDifferentiationOneArgPushforwardPrep,
    ::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        v_vec = vcat(myvec(x), myvec(dx))
        prep.jvp_exe!(vec(dy), v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::FastDifferentiationOneArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::FastDifferentiationOneArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    return f(x), DI.pushforward!(f, ty, prep, backend, x, tx)
end

## Pullback

struct FastDifferentiationOneArgPullbackPrep{E1,E1!} <: DI.PullbackPrep
    vjp_exe::E1
    vjp_exe!::E1!
end

function DI.prepare_pullback(f, ::AutoFastDifferentiation, x, ty::NTuple)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    vj_vec_var, v_vec_var = jacobian_transpose_v(y_vec_var, x_vec_var)
    vjp_exe = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    vjp_exe! = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationOneArgPullbackPrep(vjp_exe, vjp_exe!)
end

function DI.pullback(
    f, prep::FastDifferentiationOneArgPullbackPrep, ::AutoFastDifferentiation, x, ty::NTuple
)
    tx = map(ty) do dy
        v_vec = vcat(myvec(x), myvec(dy))
        if x isa Number
            return only(prep.vjp_exe(v_vec))
        else
            return reshape(prep.vjp_exe(v_vec), size(x))
        end
    end
    return tx
end

function DI.pullback!(
    f,
    tx::NTuple,
    prep::FastDifferentiationOneArgPullbackPrep,
    ::AutoFastDifferentiation,
    x,
    ty::NTuple,
)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        v_vec = vcat(myvec(x), myvec(dy))
        prep.vjp_exe!(vec(dx), v_vec)
    end
    return tx
end

function DI.value_and_pullback(
    f,
    prep::FastDifferentiationOneArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
)
    return f(x), DI.pullback(f, prep, backend, x, ty)
end

function DI.value_and_pullback!(
    f,
    tx::NTuple,
    prep::FastDifferentiationOneArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
)
    return f(x), DI.pullback!(f, tx, prep, backend, x, ty)
end

## Derivative

struct FastDifferentiationOneArgDerivativePrep{Y,E1,E1!} <: DI.DerivativePrep
    y_prototype::Y
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(f, ::AutoFastDifferentiation, x)
    y_prototype = f(x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var; in_place=false)
    der_exe! = make_function(der_vec_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgDerivativePrep(y_prototype, der_exe, der_exe!)
end

function DI.derivative(
    f, prep::FastDifferentiationOneArgDerivativePrep, ::AutoFastDifferentiation, x
)
    if prep.y_prototype isa Number
        return only(prep.der_exe(monovec(x)))
    else
        return reshape(prep.der_exe(monovec(x)), size(prep.y_prototype))
    end
end

function DI.derivative!(
    f, der, prep::FastDifferentiationOneArgDerivativePrep, ::AutoFastDifferentiation, x
)
    prep.der_exe!(vec(der), monovec(x))
    return der
end

function DI.value_and_derivative(
    f, prep::FastDifferentiationOneArgDerivativePrep, backend::AutoFastDifferentiation, x
)
    return f(x), DI.derivative(f, prep, backend, x)
end

function DI.value_and_derivative!(
    f,
    der,
    prep::FastDifferentiationOneArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
)
    return f(x), DI.derivative!(f, der, prep, backend, x)
end

## Gradient

struct FastDifferentiationOneArgGradientPrep{E1,E1!} <: DI.GradientPrep
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_gradient(f, backend::AutoFastDifferentiation, x)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    y_vec_var = monovec(y_var)
    jac_var = jacobian(y_vec_var, x_vec_var)
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgGradientPrep(jac_exe, jac_exe!)
end

function DI.gradient(
    f, prep::FastDifferentiationOneArgGradientPrep, ::AutoFastDifferentiation, x
)
    jac = prep.jac_exe(vec(x))
    grad_vec = @view jac[1, :]
    return reshape(grad_vec, size(x))
end

function DI.gradient!(
    f, grad, prep::FastDifferentiationOneArgGradientPrep, ::AutoFastDifferentiation, x
)
    prep.jac_exe!(reshape(grad, 1, length(grad)), vec(x))
    return grad
end

function DI.value_and_gradient(
    f, prep::FastDifferentiationOneArgGradientPrep, backend::AutoFastDifferentiation, x
)
    return f(x), DI.gradient(f, prep, backend, x)
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::FastDifferentiationOneArgGradientPrep,
    backend::AutoFastDifferentiation,
    x,
)
    return f(x), DI.gradient!(f, grad, prep, backend, x)
end

## Jacobian

struct FastDifferentiationOneArgJacobianPrep{Y,E1,E1!} <: DI.JacobianPrep
    y_prototype::Y
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f, backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}}, x
)
    y_prototype = f(x)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    y_vec_var = vec(y_var)
    jac_var = if backend isa AutoSparse
        sparse_jacobian(y_vec_var, x_vec_var)
    else
        jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationOneArgJacobianPrep(y_prototype, jac_exe, jac_exe!)
end

function DI.jacobian(
    f,
    prep::FastDifferentiationOneArgJacobianPrep,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return prep.jac_exe(vec(x))
end

function DI.jacobian!(
    f,
    jac,
    prep::FastDifferentiationOneArgJacobianPrep,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    prep.jac_exe!(jac, vec(x))
    return jac
end

function DI.value_and_jacobian(
    f,
    prep::FastDifferentiationOneArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return f(x), DI.jacobian(f, prep, backend, x)
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::FastDifferentiationOneArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return f(x), DI.jacobian!(f, jac, prep, backend, x)
end

## Second derivative

struct FastDifferentiationAllocatingSecondDerivativePrep{Y,D,E2,E2!} <:
       DI.SecondDerivativePrep
    y_prototype::Y
    derivative_prep::D
    der2_exe::E2
    der2_exe!::E2!
end

function DI.prepare_second_derivative(f, backend::AutoFastDifferentiation, x)
    y_prototype = f(x)
    x_var = only(make_variables(:x))
    y_var = f(x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)

    der2_vec_var = derivative(y_vec_var, x_var, x_var)
    der2_exe = make_function(der2_vec_var, x_vec_var; in_place=false)
    der2_exe! = make_function(der2_vec_var, x_vec_var; in_place=true)

    derivative_prep = DI.prepare_derivative(f, backend, x)
    return FastDifferentiationAllocatingSecondDerivativePrep(
        y_prototype, derivative_prep, der2_exe, der2_exe!
    )
end

function DI.second_derivative(
    f, prep::FastDifferentiationAllocatingSecondDerivativePrep, ::AutoFastDifferentiation, x
)
    if prep.y_prototype isa Number
        return only(prep.der2_exe(monovec(x)))
    else
        return reshape(prep.der2_exe(monovec(x)), size(prep.y_prototype))
    end
end

function DI.second_derivative!(
    f,
    der2,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
)
    prep.der2_exe!(vec(der2), monovec(x))
    return der2
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
)
    y, der = DI.value_and_derivative(f, prep.derivative_prep, backend, x)
    der2 = DI.second_derivative(f, prep, backend, x)
    return y, der, der2
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    prep::FastDifferentiationAllocatingSecondDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
)
    y, _ = DI.value_and_derivative!(f, der, prep.derivative_prep, backend, x)
    DI.second_derivative!(f, der2, prep, backend, x)
    return y, der, der2
end

## HVP

struct FastDifferentiationHVPPrep{E2,E2!,E1} <: DI.HVPPrep
    hvp_exe::E2
    hvp_exe!::E2!
    gradient_prep::E1
end

function DI.prepare_hvp(f, backend::AutoFastDifferentiation, x, tx::NTuple)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)
    hv_vec_var, v_vec_var = hessian_times_v(y_var, x_vec_var)
    hvp_exe = make_function(hv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    hvp_exe! = make_function(hv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)

    gradient_prep = DI.prepare_gradient(f, backend, x)
    return FastDifferentiationHVPPrep(hvp_exe, hvp_exe!, gradient_prep)
end

function DI.hvp(
    f, prep::FastDifferentiationHVPPrep, ::AutoFastDifferentiation, x, tx::NTuple
)
    tg = map(tx) do dx
        v_vec = vcat(vec(x), vec(dx))
        dg_vec = prep.hvp_exe(v_vec)
        return reshape(dg_vec, size(x))
    end
    return tg
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::FastDifferentiationHVPPrep,
    ::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    for b in eachindex(tx, tg)
        dx, dg = tx[b], tg[b]
        v_vec = vcat(vec(x), vec(dx))
        prep.hvp_exe!(dg, v_vec)
    end
    return tg
end

function DI.gradient_and_hvp(
    f, prep::FastDifferentiationHVPPrep, backend::AutoFastDifferentiation, x, tx::NTuple
)
    tg = DI.hvp(f, prep, backend, x, tx)
    grad = DI.gradient(f, prep.gradient_prep, backend, x)
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
)
    DI.hvp!(f, tg, prep, backend, x, tx)
    DI.gradient!(f, grad, prep.gradient_prep, backend, x)
    return grad, tg
end

## Hessian

struct FastDifferentiationHessianPrep{G,E2,E2!} <: DI.HessianPrep
    gradient_prep::G
    hess_exe::E2
    hess_exe!::E2!
end

function DI.prepare_hessian(
    f, backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}}, x
)
    x_var = make_variables(:x, size(x)...)
    y_var = f(x_var)

    x_vec_var = vec(x_var)

    hess_var = if backend isa AutoSparse
        sparse_hessian(y_var, x_vec_var)
    else
        hessian(y_var, x_vec_var)
    end
    hess_exe = make_function(hess_var, x_vec_var; in_place=false)
    hess_exe! = make_function(hess_var, x_vec_var; in_place=true)

    gradient_prep = DI.prepare_gradient(f, dense_ad(backend), x)
    return FastDifferentiationHessianPrep(gradient_prep, hess_exe, hess_exe!)
end

function DI.hessian(
    f,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    return prep.hess_exe(vec(x))
end

function DI.hessian!(
    f,
    hess,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    prep.hess_exe!(hess, vec(x))
    return hess
end

function DI.value_gradient_and_hessian(
    f,
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
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
    prep::FastDifferentiationHessianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    y, _ = DI.value_and_gradient!(f, grad, prep.gradient_prep, dense_ad(backend), x)
    DI.hessian!(f, hess, prep, backend, x)
    return y, grad, hess
end
