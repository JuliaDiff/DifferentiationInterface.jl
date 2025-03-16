## Pushforward

struct SymbolicsTwoArgPushforwardPrep{E1,E1!} <: DI.PushforwardPrep
    pushforward_exe::E1
    pushforward_exe!::E1!
end

function DI.prepare_pushforward(
    f!, y, backend::AutoSymbolics, x, tx::NTuple, contexts::Vararg{DI.Context,C}
) where {C}
    dx = first(tx)
    x_var = variablize(x, :x)
    dx_var = variablize(dx, :dx)
    context_vars = variablize(contexts)
    y_var = variablize(y, :y)
    t_var = variable(:t)
    f!(y_var, x_var + t_var * dx_var, context_vars...)
    step_der_var = derivative(y_var, t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    res = build_function(pf_var, x_var, dx_var, context_vars...; expression=Val(false))
    (pushforward_exe, pushforward_exe!) = res
    return SymbolicsTwoArgPushforwardPrep(pushforward_exe, pushforward_exe!)
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
    DI.pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, ty
end

## Derivative

struct SymbolicsTwoArgDerivativePrep{E1,E1!} <: DI.DerivativePrep
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(
    f!, y, backend::AutoSymbolics, x, contexts::Vararg{DI.Context,C}
) where {C}
    x_var = variablize(x, :x)
    y_var = variablize(y, :y)
    context_vars = variablize(contexts)
    f!(y_var, x_var, context_vars...)
    der_var = derivative(y_var, x_var)

    res = build_function(der_var, x_var, context_vars...; expression=Val(false))
    (der_exe, der_exe!) = res
    return SymbolicsTwoArgDerivativePrep(der_exe, der_exe!)
end

function DI.derivative(
    f!,
    y,
    prep::SymbolicsTwoArgDerivativePrep,
    backend::AutoSymbolics,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    DI.derivative!(f!, y, der, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, der
end

## Jacobian

struct SymbolicsTwoArgJacobianPrep{E1,E1!} <: DI.JacobianPrep
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f!,
    y,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    x_var = variablize(x, :x)
    y_var = variablize(y, :y)
    context_vars = variablize(contexts)
    f!(y_var, x_var, context_vars...)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(y_var), vec(x_var))
    else
        jacobian(y_var, x_var)
    end

    res = build_function(jac_var, x_var, context_vars...; expression=Val(false))
    (jac_exe, jac_exe!) = res
    return SymbolicsTwoArgJacobianPrep(jac_exe, jac_exe!)
end

function DI.jacobian(
    f!,
    y,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
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
    DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
    f!(y, x, map(DI.unwrap, contexts)...)
    return y, jac
end
