## Pushforward

struct FastDifferentiationTwoArgPushforwardPrep{SIG,E1,E1!} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    jvp_exe::E1
    jvp_exe!::E1!
end

function DI.prepare_pushforward(
    strict::Val,
    f!,
    y,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = variablize(y, :y)
    f!(y_var, x_var, context_vars...)

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
    return FastDifferentiationTwoArgPushforwardPrep(_sig, jvp_exe, jvp_exe!)
end

function DI.pushforward(
    f!,
    y,
    prep::FastDifferentiationTwoArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = map(tx) do dx
        reshape(prep.jvp_exe(myvec(x), myvec(dx), map(myvec_unwrap, contexts)...), size(y))
    end
    return ty
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::FastDifferentiationTwoArgPushforwardPrep,
    backend::AutoFastDifferentiation,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        prep.jvp_exe!(myvec(dy), myvec(x), myvec(dx), map(myvec_unwrap, contexts)...)
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
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = DI.pushforward(f!, y, prep, backend, x, tx, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
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
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, ty
end

## Pullback

struct FastDifferentiationTwoArgPullbackPrep{SIG,E1,E1!} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    vjp_exe::E1
    vjp_exe!::E1!
end

function DI.prepare_pullback(
    strict::Val,
    f!,
    y,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, ty, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = variablize(y, :y)
    f!(y_var, x_var, context_vars...)

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
    return FastDifferentiationTwoArgPullbackPrep(_sig, vjp_exe, vjp_exe!)
end

function DI.pullback(
    f!,
    y,
    prep::FastDifferentiationTwoArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
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
    f!,
    y,
    tx::NTuple,
    prep::FastDifferentiationTwoArgPullbackPrep,
    backend::AutoFastDifferentiation,
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        prep.vjp_exe!(myvec(dx), myvec(x), myvec(dy), map(myvec_unwrap, contexts)...)
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
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    tx = DI.pullback(f!, y, prep, backend, x, ty, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
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
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    DI.pullback!(f!, y, tx, prep, backend, x, ty, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, tx
end

## Derivative

struct FastDifferentiationTwoArgDerivativePrep{SIG,E1,E1!} <: DI.DerivativePrep{SIG}
    _sig::Val{SIG}
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(
    strict::Val, f!, y, backend::AutoFastDifferentiation, x, contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = variablize(y, :y)
    f!(y_var, x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var, context_vec_vars...; in_place=false)
    der_exe! = make_function(der_vec_var, x_vec_var, context_vec_vars...; in_place=true)
    return FastDifferentiationTwoArgDerivativePrep(_sig, der_exe, der_exe!)
end

function DI.value_and_derivative(
    f!,
    y,
    prep::FastDifferentiationTwoArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    der = reshape(prep.der_exe(myvec(x), map(myvec_unwrap, contexts)...), size(y))
    return y, der
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::FastDifferentiationTwoArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    prep.der_exe!(myvec(der), myvec(x), map(myvec_unwrap, contexts)...)
    return y, der
end

function DI.derivative(
    f!,
    y,
    prep::FastDifferentiationTwoArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    der = reshape(prep.der_exe(myvec(x), map(myvec_unwrap, contexts)...), size(y))
    return der
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::FastDifferentiationTwoArgDerivativePrep,
    backend::AutoFastDifferentiation,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    prep.der_exe!(myvec(der), myvec(x), map(myvec_unwrap, contexts)...)
    return der
end

## Jacobian

struct FastDifferentiationTwoArgJacobianPrep{SIG,E1,E1!} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    strict::Val,
    f!,
    y,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    context_vars = variablize(contexts)
    y_var = variablize(y, :y)
    f!(y_var, x_var, context_vars...)

    x_vec_var = myvec(x_var)
    context_vec_vars = map(myvec, context_vars)
    y_vec_var = myvec(y_var)
    jac_var = if backend isa AutoSparse
        sparse_jacobian(y_vec_var, x_vec_var)
    else
        jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var, context_vec_vars...; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var, context_vec_vars...; in_place=true)
    return FastDifferentiationTwoArgJacobianPrep(_sig, jac_exe, jac_exe!)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::FastDifferentiationTwoArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    jac = prep.jac_exe(myvec(x), map(myvec_unwrap, contexts)...)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::FastDifferentiationTwoArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    prep.jac_exe!(jac, myvec(x), map(myvec_unwrap, contexts)...)
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    prep::FastDifferentiationTwoArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    jac = prep.jac_exe(myvec(x), map(myvec_unwrap, contexts)...)
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::FastDifferentiationTwoArgJacobianPrep,
    backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    prep.jac_exe!(jac, myvec(x), map(myvec_unwrap, contexts)...)
    return jac
end
