## Pushforward

struct SymbolicsTwoArgPushforwardPrep{SIG,E1,E1!} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    pushforward_exe::E1
    pushforward_exe!::E1!
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f!,
    y,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    dx = first(tx)
    x_var = variablize(x, :x)
    dx_var = variablize(dx, :dx)
    context_vars = variablize(contexts)
    y_var = variablize(y, :y)
    t_var = variable(:t)
    f!(y_var, x_var + t_var * dx_var, context_vars...)
    step_der_var = derivative(y_var, t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    erase_cache_vars!(context_vars, contexts)
    res = build_function(
        pf_var, x_var, dx_var, context_vars...; expression=Val(false), cse=true
    )
    (pushforward_exe, pushforward_exe!) = res
    return SymbolicsTwoArgPushforwardPrep(_sig, pushforward_exe, pushforward_exe!)
end

function DI.pushforward(
    f!,
    y,
    prep::SymbolicsTwoArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = map(tx) do dx
        dy = prep.pushforward_exe(x, dx, map(DI.unwrap, contexts)...)
    end
    return ty
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::SymbolicsTwoArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        prep.pushforward_exe!(dy, x, dx, map(DI.unwrap, contexts)...)
    end
    return ty
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::SymbolicsTwoArgPushforwardPrep,
    backend::AutoSymbolics,
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
    prep::SymbolicsTwoArgPushforwardPrep,
    backend::AutoSymbolics,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, ty
end

## Derivative

struct SymbolicsTwoArgDerivativePrep{SIG,E1,E1!} <: DI.DerivativePrep{SIG}
    _sig::Val{SIG}
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative_nokwarg(
    strict::Val, f!, y, backend::AutoSymbolics, x, contexts::Vararg{DI.Context,C}
) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    y_var = variablize(y, :y)
    context_vars = variablize(contexts)
    f!(y_var, x_var, context_vars...)
    der_var = derivative(y_var, x_var)

    erase_cache_vars!(context_vars, contexts)
    res = build_function(der_var, x_var, context_vars...; expression=Val(false), cse=true)
    (der_exe, der_exe!) = res
    return SymbolicsTwoArgDerivativePrep(_sig, der_exe, der_exe!)
end

function DI.derivative(
    f!,
    y,
    prep::SymbolicsTwoArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    return prep.der_exe(x, map(DI.unwrap, contexts)...)
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::SymbolicsTwoArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    prep.der_exe!(der, x, map(DI.unwrap, contexts)...)
    return der
end

function DI.value_and_derivative(
    f!,
    y,
    prep::SymbolicsTwoArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    der = DI.derivative(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, der
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::SymbolicsTwoArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    DI.derivative!(f!, y, der, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, der
end

## Jacobian

struct SymbolicsTwoArgJacobianPrep{SIG,E1,E1!} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian_nokwarg(
    strict::Val,
    f!,
    y,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C};
) where {C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    x_var = variablize(x, :x)
    y_var = variablize(y, :y)
    context_vars = variablize(contexts)
    f!(y_var, x_var, context_vars...)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(y_var), vec(x_var))
    else
        jacobian(y_var, x_var)
    end

    erase_cache_vars!(context_vars, contexts)
    res = build_function(jac_var, x_var, context_vars...; expression=Val(false), cse=true)
    (jac_exe, jac_exe!) = res
    return SymbolicsTwoArgJacobianPrep(_sig, jac_exe, jac_exe!)
end

function DI.jacobian(
    f!,
    y,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    return prep.jac_exe(x, map(DI.unwrap, contexts)...)
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    prep.jac_exe!(jac, x, map(DI.unwrap, contexts)...)
    return jac
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    jac = DI.jacobian(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, jac
end
