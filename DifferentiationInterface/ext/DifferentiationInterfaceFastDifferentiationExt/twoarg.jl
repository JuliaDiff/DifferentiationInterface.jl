## Pushforward

struct FastDifferentiationTwoArgPushforwardPrep{E1,E1!} <: DI.PushforwardPrep
    jvp_exe::E1
    jvp_exe!::E1!
end

function DI.prepare_pushforward(f!, y, ::AutoFastDifferentiation, x, tx::NTuple)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    jvp_exe! = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationTwoArgPushforwardPrep(jvp_exe, jvp_exe!)
end

function DI.pushforward(
    f!,
    y,
    prep::FastDifferentiationTwoArgPushforwardPrep,
    ::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    ty = map(tx) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        reshape(prep.jvp_exe(v_vec), size(y))
    end
    return ty
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::FastDifferentiationTwoArgPushforwardPrep,
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
    f!,
    y,
    prep::FastDifferentiationTwoArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    ty = DI.pushforward(f!, y, prep, backend, x, tx)
    f!(y, x)
    return y, ty
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::FastDifferentiationTwoArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
)
    DI.pushforward!(f!, y, ty, prep, backend, x, tx)
    f!(y, x)
    return y, ty
end

## Pullback

struct FastDifferentiationTwoArgPullbackPrep{E1,E1!} <: DI.PullbackPrep
    vjp_exe::E1
    vjp_exe!::E1!
end

function DI.prepare_pullback(f!, y, ::AutoFastDifferentiation, x, ty::NTuple)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    vj_vec_var, v_vec_var = jacobian_transpose_v(y_vec_var, x_vec_var)
    vjp_exe = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    vjp_exe! = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationTwoArgPullbackPrep(vjp_exe, vjp_exe!)
end

function DI.pullback(
    f!,
    y,
    prep::FastDifferentiationTwoArgPullbackPrep,
    ::AutoFastDifferentiation,
    x,
    ty::NTuple,
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
    f!,
    y,
    tx::NTuple,
    prep::FastDifferentiationTwoArgPullbackPrep,
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
    f!,
    y,
    prep::FastDifferentiationTwoArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
)
    tx = DI.pullback(f!, y, prep, backend, x, ty)
    f!(y, x)
    return y, tx
end

function DI.value_and_pullback!(
    f!,
    y,
    tx::NTuple,
    prep::FastDifferentiationTwoArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
)
    DI.pullback!(f!, y, tx, prep, backend, x, ty)
    f!(y, x)
    return y, tx
end

## Derivative

struct FastDifferentiationTwoArgDerivativePrep{E1,E1!} <: DI.DerivativePrep
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(f!, y, ::AutoFastDifferentiation, x)
    x_var = only(make_variables(:x))
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var; in_place=false)
    der_exe! = make_function(der_vec_var, x_vec_var; in_place=true)
    return FastDifferentiationTwoArgDerivativePrep(der_exe, der_exe!)
end

function DI.value_and_derivative(
    f!, y, prep::FastDifferentiationTwoArgDerivativePrep, ::AutoFastDifferentiation, x
)
    f!(y, x)
    der = reshape(prep.der_exe(monovec(x)), size(y))
    return y, der
end

function DI.value_and_derivative!(
    f!, y, der, prep::FastDifferentiationTwoArgDerivativePrep, ::AutoFastDifferentiation, x
)
    f!(y, x)
    prep.der_exe!(der, monovec(x))
    return y, der
end

function DI.derivative(
    f!, y, prep::FastDifferentiationTwoArgDerivativePrep, ::AutoFastDifferentiation, x
)
    der = reshape(prep.der_exe(monovec(x)), size(y))
    return der
end

function DI.derivative!(
    f!, y, der, prep::FastDifferentiationTwoArgDerivativePrep, ::AutoFastDifferentiation, x
)
    prep.der_exe!(der, monovec(x))
    return der
end

## Jacobian

struct FastDifferentiationTwoArgJacobianPrep{E1,E1!} <: DI.JacobianPrep
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f!, y, backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}}, x
)
    x_var = make_variables(:x, size(x)...)
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = vec(x_var)
    y_vec_var = vec(y_var)
    jac_var = if backend isa AutoSparse
        sparse_jacobian(y_vec_var, x_vec_var)
    else
        jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationTwoArgJacobianPrep(jac_exe, jac_exe!)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::FastDifferentiationTwoArgJacobianPrep,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    f!(y, x)
    jac = prep.jac_exe(vec(x))
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::FastDifferentiationTwoArgJacobianPrep,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    f!(y, x)
    prep.jac_exe!(jac, vec(x))
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    prep::FastDifferentiationTwoArgJacobianPrep,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    jac = prep.jac_exe(vec(x))
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::FastDifferentiationTwoArgJacobianPrep,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
)
    prep.jac_exe!(jac, vec(x))
    return jac
end
