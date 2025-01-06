## Pushforward

# Contains either a single pre-allocated initial TPS
# or a vector of pre-allocated TPSs.
struct GTPSAOneArgPushforwardPrep{X} <: DI.PushforwardPrep
    xt::X
end

function DI.prepare_pushforward(
    ::F, 
    backend::AutoGTPSA{D}, 
    x, 
    tx::NTuple,
    ::Vararg{DI.Context,C},
) where {F,D,C}
    
    # For pushforward/JVP, we only actually need 1 single variable (in the GTPSA sense)
    # because we even if we did multiple we will add up the derivatives of each at the end.
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 1) # 1 variable to first order
    end
    if x isa Number
        xt = TPS{promote_type(typeof(first(tx)), typeof(x), Float64)}(; use=d)
        return GTPSAOneArgPushforwardPrep(xt)
    else
        xt = similar(x, TPS{promote_type(eltype(first(tx)), eltype(x), Float64)})
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(first(tx)), eltype(x), Float64)}(; use=d)
        end
        return GTPSAOneArgPushforwardPrep(xt)
    end
end

function DI.pushforward(
    f, 
    prep::GTPSAOneArgPushforwardPrep, 
    ::AutoGTPSA, 
    x, 
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    ty = map(tx) do dx
        initialize!(prep.xt, x, dx, 1)
        yt = fc(prep.xt)
        if yt isa Number
            return yt[1]
        else
            dy = map(t -> t[1], yt)
            return dy
        end
    end
    return ty
end

function DI.pushforward!(
    f, 
    ty::NTuple, 
    prep::GTPSAOneArgPushforwardPrep, 
    ::AutoGTPSA, 
    x, 
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        initialize!(prep.xt, x, dx, 1)
        yt = fc(prep.xt)
        map!(t -> t[1], dy, yt)
    end
    return ty
end

function DI.value_and_pushforward(
    f, 
    prep::GTPSAOneArgPushforwardPrep, 
    backend::AutoGTPSA, 
    x, 
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}

    fc = DI.with_contexts(f, contexts...)
    ty = DI.pushforward(fc, prep, backend, x, tx)
    y = fc(x) # This actually seems faster than indexing the TPSs
    return y, ty
end

function DI.value_and_pushforward!(
    f, 
    ty::NTuple, 
    prep::GTPSAOneArgPushforwardPrep, 
    ::AutoGTPSA, 
    x, 
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    fc = DI.with_contexts(f, contexts...)
    DI.pushforward!(fc, ty, prep, backend, x, tx)
    y = fc(x)
    return y, ty
end

## Gradient
# Contains either a single pre-allocated initial TPS
# or a vector of pre-allocated TPSs.
struct GTPSAOneArgGradientPrep{X} <: DI.GradientPrep
    xt::X
end

# Unlike JVP, this requires us to use all variables 
function DI.prepare_gradient(
    f,
    backend::AutoGTPSA{D},
    x,
    contexts::Vararg{DI.Context,C},
) where {D,C}

    if D != Nothing
        d = backend.descriptor
        length(x) == GTPSA.numnn(d) || error("Number of variables + parameters in Descriptor disagrees with function number of inputs!")
    else
        d = Descriptor(length(x), 1) # n variables to first order
    end

    # We set the slopes of each variable to 1 here, this will always be the case for gradient
    if x isa Number
        xt = TPS{promote_type(typeof(x), Float64)}(; use=d)
        xt[1] = 1
        return GTPSAOneArgGradientPrep(xt)
    else
        xt = similar(x, TPS{promote_type(eltype(x), Float64)})
        j = 1
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(x), Float64)}(; use=d)
            xt[i][j] = 1
            j += 1
        end
        return GTPSAOneArgGradientPrep(xt)
    end
    return GTPSAOneArgGradientPrep(xt)
end

function DI.gradient(
    f, 
    prep::GTPSAOneArgGradientPrep, 
    ::AutoGTPSA, 
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part (slopes set in prepare)
    fc = DI.with_contexts(f, contexts...)
    yt = fc(prep.xt)
    if prep.xt isa Number
        return yt[1]
    else
        grad = similar(x, GTPSA.numtype(eltype(yt)))
        GTPSA.gradient!(grad, yt; include_params=true)
    end
    return grad
end

function DI.gradient!(
    f, 
    grad,
    prep::GTPSAOneArgGradientPrep, 
    ::AutoGTPSA, 
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    GTPSA.gradient!(grad, yt; include_params=true)
    return grad
end

#=
function DI.value_and_gradient(f, prep::GTPSAGradientPrep, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

function DI.value_and_gradient!(f, grad, prep::GTPSAGradientPrep, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    GTPSA.gradient!(grad, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, grad
end

## Jacobian

struct GTPSAJacobianprep{X} <: Jacobianprep
    xt::X
end

function DI.prepare_jacobian(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
    else
        nn = length(x)
        d = Descriptor(nn, 1)
    end
    xt = similar(x, TPS{promote_type(eltype(x), Float64)})

    # xt and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(x), Float64)}(; use=d)
        xt[i][j] = 1
        j += 1
    end

    return GTPSAJacobianprep(xt)
end

function DI.jacobian(f, prep::GTPSAJacobianprep, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    jac = similar(x, eltype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.jacobian!(f, jac, prep::GTPSAJacobianprep, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    GTPSA.jacobian!(jac, yt; include_params=true)
    return jac
end

function DI.value_and_jacobian(f, prep::GTPSAJacobianprep, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    jac = similar(x, eltype(eltype(yt)), (length(yt), length(x)))
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

function DI.value_and_jacobian!(f, jac, prep::GTPSAJacobianprep, ::AutoGTPSA, x)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    GTPSA.jacobian!(jac, yt; include_params=true)
    y = map(t -> t[0], yt)
    return y, jac
end

## Second derivative

struct GTPSASecondDerivativeprep{X} <: SecondDerivativeprep
    xt::X
end

function DI.prepare_second_derivative(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 2)
    end
    xt = TPS{promote_type(typeof(x), Float64)}(; use=d)
    xt[1] = 1
    return GTPSASecondDerivativeprep(xt)
end

function DI.second_derivative(f, prep::GTPSASecondDerivativeprep, ::AutoGTPSA, x)
    prep.xt[0] = x
    yt = f(prep.xt)
    if yt isa Number
        return yt[2] * 2
    else
        der2 = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der2[i] = yt[i][2] * 2 # *2 because monomial coefficient is 1/2
        end
        return der2
    end
end

function DI.second_derivative!(f, der2, prep::GTPSASecondDerivativeprep, ::AutoGTPSA, x)
    prep.xt[0] = x
    yt = f(prep.xt)
    for i in eachindex(yt)
        der2[i] = yt[i][2] * 2
    end
    return der2
end

function DI.value_derivative_and_second_derivative(
    f, prep::GTPSASecondDerivativeprep, ::AutoGTPSA, x
)
    prep.xt[0] = x
    yt = f(prep.xt)
    if yt isa Number
        return yt[0], yt[1], yt[2] * 2
    else
        y = map(t -> t[0], yt)
        der = similar(yt, eltype(eltype(yt)))
        der2 = similar(yt, eltype(eltype(yt)))
        for i in eachindex(yt)
            der[i] = yt[i][1]
            der2[i] = yt[i][2] * 2
        end
        return y, der, der2
    end
end

function DI.value_derivative_and_second_derivative!(
    f, der, der2, prep::GTPSASecondDerivativeprep, ::AutoGTPSA, x
)
    prep.xt[0] = x
    yt = f(prep.xt)
    y = map(t -> t[0], yt)
    for i in eachindex(yt)
        der[i] = yt[i][1]
        der2[i] = yt[i][2] * 2
    end
    return y, der, der2
end

## Hessian

struct GTPSAHessianprep{X,M} <: Hessianprep
    xt::X
    m::M
end

function DI.prepare_hessian(f, backend::AutoGTPSA{D}, x) where {D}
    if D != Nothing
        d = backend.descriptor
        nn = GTPSA.numnn(d)
        m = Vector{UInt8}(undef, nn)
    else
        nn = length(x)
        d = Descriptor(nn, 2)
        # If all variables/variable+parameters have truncation order > 2, then 
        # the indexing is known beforehand and we can do it (very slightly) faster
        m = nothing
    end
    xt = similar(x, TPS{promote_type(eltype(x), Float64)})

    # xt and x have same indexing because of similar
    # Setting the first derivatives must be 1-based 
    # linear with the variables.
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(x), Float64)}(; use=d)
        xt[i][j] = 1
        j += 1
    end

    return GTPSAHessianprep(xt, m)
end

function DI.hessian(f, prep::GTPSAHessianprep, ::AutoGTPSA{D}, x) where {D}
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    hess = similar(x, eltype(eltype(yt)), (length(x), length(x)))
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=prep.m
    )
    return hess
end

function DI.hessian!(f, hess, prep::GTPSAHessianprep, ::AutoGTPSA{D}, x) where {D}
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=prep.m
    )
    return hess
end

function DI.value_gradient_and_hessian(
    f, prep::GTPSAHessianprep, ::AutoGTPSA{D}, x
) where {D}
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    y = map(t -> t[0], yt)
    grad = similar(x, eltype(eltype(yt)))
    GTPSA.gradient!(grad, yt; include_params=true)
    hess = similar(x, eltype(eltype(yt)), (length(x), length(x)))
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=prep.m
    )
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, prep::GTPSAHessianprep, ::AutoGTPSA{D}, x
) where {D}
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    yt = f(prep.xt)
    y = map(t -> t[0], yt)
    GTPSA.gradient!(grad, yt; include_params=true)
    unsafe_fast = D == Nothing ? true : false
    GTPSA.hessian!(
        hess, yt; include_params=true, unsafe_fast=unsafe_fast, tmp_mono=prep.m
    )
    return y, grad, hess
end

# HVP

struct GTPSAHVPprep{E,H} <: HVPprep
    hessprep::E
    hess::H
end

function DI.prepare_hvp(f, backend::AutoGTPSA{D}, x, tx::Tangents) where {D}
    hessprep = DI.prepare_hessian(f, backend, x)
    hess = similar(x, typeof(f(x)), (length(x), length(x)))
    return GTPSAHVPprep(hessprep, hess)
end

function DI.hvp(f, prep::GTPSAHVPprep, backend::AutoGTPSA{D}, x, tx::Tangents) where {D}
    DI.hessian!(f, prep.hess, prep.hessprep, backend, x)
    tg = map(tx) do dx
        dg = similar(x, eltype(prep.hess))
        dg .= 0
        j = 1
        for dxi in dx
            for i in 1:size(prep.hess, 2)
                dg[i] += prep.hess[i, j] * dxi
            end
            j += 1
        end
        return dg
    end
    return tg
end

function DI.hvp!(
    f, tg::Tangents, prep::GTPSAHVPprep, backend::AutoGTPSA{D}, x, tx::Tangents
) where {D}
    DI.hessian!(f, prep.hess, prep.hessprep, backend, x)
    for b in eachindex(tg.d)
        dx, dg = tx.d[b], tg.d[b]
        dg .= 0
        j = 1
        for dxi in dx
            for i in 1:size(prep.hess, 2)
                dg[i] += prep.hess[i, j] * dxi
            end
            j += 1
        end
    end
    return tg
end
=#