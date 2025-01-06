## Pushforward

struct SymbolicsTwoArgPushforwardPrep{E1,E1!} <: DI.PushforwardPrep
    pushforward_exe::E1
    pushforward_exe!::E1!
end

function DI.prepare_pushforward(f!, y, ::AutoSymbolics, x, tx::NTuple)
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
    y_var = variables(:y, axes(y)...)
    t_var = variable(:t)
    f!(y_var, x_var + t_var * dx_var)
    step_der_var = derivative(y_var, t_var)
    pf_var = substitute(step_der_var, Dict(t_var => zero(eltype(x))))

    res = build_function(pf_var, vcat(myvec(x_var), myvec(dx_var)); expression=Val(false))
    (pushforward_exe, pushforward_exe!) = res
    return SymbolicsTwoArgPushforwardPrep(pushforward_exe, pushforward_exe!)
end

function DI.pushforward(
    f!, y, prep::SymbolicsTwoArgPushforwardPrep, ::AutoSymbolics, x, tx::NTuple
)
    ty = map(tx) do dx
        v_vec = vcat(myvec(x), myvec(dx))
        dy = prep.pushforward_exe(v_vec)
    end
    return ty
end

function DI.pushforward!(
    f!, y, ty::NTuple, prep::SymbolicsTwoArgPushforwardPrep, ::AutoSymbolics, x, tx::NTuple
)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        v_vec = vcat(myvec(x), myvec(dx))
        prep.pushforward_exe!(dy, v_vec)
    end
    return ty
end

function DI.value_and_pushforward(
    f!, y, prep::SymbolicsTwoArgPushforwardPrep, backend::AutoSymbolics, x, tx::NTuple
)
    ty = DI.pushforward(f!, y, prep, backend, x, tx)
    f!(y, x)
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
)
    DI.pushforward!(f!, y, ty, prep, backend, x, tx)
    f!(y, x)
    return y, ty
end

## Derivative

struct SymbolicsTwoArgDerivativePrep{E1,E1!} <: DI.DerivativePrep
    der_exe::E1
    der_exe!::E1!
end

function DI.prepare_derivative(f!, y, ::AutoSymbolics, x)
    x_var = variable(:x)
    y_var = variables(:y, axes(y)...)
    f!(y_var, x_var)
    der_var = derivative(y_var, x_var)

    res = build_function(der_var, x_var; expression=Val(false))
    (der_exe, der_exe!) = res
    return SymbolicsTwoArgDerivativePrep(der_exe, der_exe!)
end

function DI.derivative(f!, y, prep::SymbolicsTwoArgDerivativePrep, ::AutoSymbolics, x)
    return prep.der_exe(x)
end

function DI.derivative!(f!, y, der, prep::SymbolicsTwoArgDerivativePrep, ::AutoSymbolics, x)
    prep.der_exe!(der, x)
    return der
end

function DI.value_and_derivative(
    f!, y, prep::SymbolicsTwoArgDerivativePrep, backend::AutoSymbolics, x
)
    der = DI.derivative(f!, y, prep, backend, x)
    f!(y, x)
    return y, der
end

function DI.value_and_derivative!(
    f!, y, der, prep::SymbolicsTwoArgDerivativePrep, backend::AutoSymbolics, x
)
    DI.derivative!(f!, y, der, prep, backend, x)
    f!(y, x)
    return y, der
end

## Jacobian

struct SymbolicsTwoArgJacobianPrep{E1,E1!} <: DI.JacobianPrep
    jac_exe::E1
    jac_exe!::E1!
end

function DI.prepare_jacobian(
    f!, y, backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}}, x
)
    x_var = variables(:x, axes(x)...)
    y_var = variables(:y, axes(y)...)
    f!(y_var, x_var)
    jac_var = if backend isa AutoSparse
        sparsejacobian(vec(y_var), vec(x_var))
    else
        jacobian(y_var, x_var)
    end

    res = build_function(jac_var, x_var; expression=Val(false))
    (jac_exe, jac_exe!) = res
    return SymbolicsTwoArgJacobianPrep(jac_exe, jac_exe!)
end

function DI.jacobian(
    f!,
    y,
    prep::SymbolicsTwoArgJacobianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    return prep.jac_exe(x)
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::SymbolicsTwoArgJacobianPrep,
    ::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    prep.jac_exe!(jac, x)
    return jac
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    jac = DI.jacobian(f!, y, prep, backend, x)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::SymbolicsTwoArgJacobianPrep,
    backend::Union{AutoSymbolics,AutoSparse{<:AutoSymbolics}},
    x,
)
    DI.jacobian!(f!, y, jac, prep, backend, x)
    f!(y, x)
    return y, jac
end
