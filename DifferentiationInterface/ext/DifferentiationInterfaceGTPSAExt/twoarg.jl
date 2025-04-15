## Pushforward

# Input: Contains either a single pre-allocated initial TPS
# or a vector of pre-allocated TPSs.
#
# Output: Contains a vector of pre-allocated TPSs
struct GTPSATwoArgPushforwardPrep{SIG,X,Y} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    xt::X
    yt::Y
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f!::F,
    y,
    backend::AutoGTPSA{D},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Constant,C};
) where {F,D,C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    # For pushforward/JVP, we only actually need 1 single variable (in the GTPSA sense)
    # because we even if we did multiple we will add up the derivatives of each at the end.
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(1, 1) # 1 variable to first order
    end
    if x isa Number
        xt = TPS{promote_type(typeof(first(tx)), typeof(x), Float64)}(; use=d)
    else
        xt = similar(x, TPS{promote_type(eltype(first(tx)), eltype(x), Float64)})
        for i in eachindex(xt)
            xt[i] = TPS{promote_type(eltype(first(tx)), eltype(x), Float64)}(; use=d)
        end
    end

    yt = similar(y, TPS{promote_type(eltype(y), Float64)})

    for i in eachindex(yt)
        yt[i] = TPS{promote_type(eltype(y), Float64)}(; use=d)
    end
    return GTPSATwoArgPushforwardPrep(_sig, xt, yt)
end

function DI.pushforward(
    f!,
    y,
    prep::GTPSATwoArgPushforwardPrep,
    backend::AutoGTPSA,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    ty = map(tx) do dx
        foreach((t, xi, dxi) -> (t[0] = xi; t[1] = dxi), prep.xt, x, dx)
        fc!(prep.yt, prep.xt)
        dy = map(t -> t[1], prep.yt)
        return dy
    end
    map!(t -> t[0], y, prep.yt)
    return ty
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::GTPSATwoArgPushforwardPrep,
    backend::AutoGTPSA,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    for b in eachindex(tx, ty)
        dx, dy = tx[b], ty[b]
        foreach((t, xi, dxi) -> (t[0] = xi; t[1] = dxi), prep.xt, x, dx)
        fc!(prep.yt, prep.xt)
        map!(t -> t[1], dy, prep.yt)
    end
    map!(t -> t[0], y, prep.yt)
    return ty
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::GTPSATwoArgPushforwardPrep,
    backend::AutoGTPSA,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = DI.pushforward(f!, y, prep, backend, x, tx, contexts...)
    return y, ty
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::GTPSATwoArgPushforwardPrep,
    backend::AutoGTPSA,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    return y, ty
end

## Jacobian
# Input: Contains a vector of pre-allocated TPSs
# Output: Contains a vector of pre-allocated TPSs
struct GTPSATwoArgJacobianPrep{SIG,X,Y} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    xt::X
    yt::Y
end

function DI.prepare_jacobian_nokwarg(
    strict::Val, f!, y, backend::AutoGTPSA{D}, x, contexts::Vararg{DI.Constant,C}
) where {D,C}
    _sig = DI.signature(f!, y, backend, x, contexts...; strict)
    if D != Nothing
        d = backend.descriptor
    else
        d = Descriptor(length(x), 1) # n variables to first order
    end

    # We set the slopes of each variable to 1 here, this will always be the case for Jacobian
    xt = similar(x, TPS{promote_type(eltype(x), Float64)})
    j = 1
    for i in eachindex(xt)
        xt[i] = TPS{promote_type(eltype(x), Float64)}(; use=d)
        xt[i][j] = 1
        j += 1
    end

    yt = similar(y, TPS{promote_type(eltype(y), Float64)})

    for i in eachindex(yt)
        yt[i] = TPS{promote_type(eltype(y), Float64)}(; use=d)
    end

    return GTPSATwoArgJacobianPrep(_sig, xt, yt)
end

function DI.jacobian(
    f!,
    y,
    prep::GTPSATwoArgJacobianPrep,
    backend::AutoGTPSA,
    x,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(prep.yt, prep.xt)
    jac = similar(x, GTPSA.numtype(eltype(prep.yt)), (length(prep.yt), length(x)))
    GTPSA.jacobian!(jac, prep.yt; include_params=true, unsafe_inbounds=true)
    map!(t -> t[0], y, prep.yt)
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::GTPSATwoArgJacobianPrep,
    backend::AutoGTPSA,
    x,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    foreach((t, xi) -> t[0] = xi, prep.xt, x) # Set the scalar part
    fc! = DI.fix_tail(f!, map(DI.unwrap, contexts)...)
    fc!(prep.yt, prep.xt)
    GTPSA.jacobian!(jac, prep.yt; include_params=true, unsafe_inbounds=true)
    map!(t -> t[0], y, prep.yt)
    return jac
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::GTPSATwoArgJacobianPrep,
    backend::AutoGTPSA,
    x,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    jac = DI.jacobian(f!, y, prep, backend, x, contexts...) # y set on line 151
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::GTPSATwoArgJacobianPrep,
    backend::AutoGTPSA,
    x,
    contexts::Vararg{DI.Constant,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
    return y, jac
end
