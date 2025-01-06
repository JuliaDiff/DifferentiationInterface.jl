## Pushforward

function DI.prepare_pushforward(
    f::F,
    ::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.NoPushforwardPrep()
end

function DI.value_and_pushforward(
    f::F,
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    dx_sametype = convert(typeof(x), only(tx))
    x_and_dx = Duplicated(x, dx_sametype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dy, y = autodiff(mode, f_and_df, x_and_dx, annotated_contexts...)
    return y, (dy,)
end

function DI.value_and_pushforward(
    f::F,
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    ty, y = autodiff(mode, f_and_df, x_and_tx, annotated_contexts...)
    return y, values(ty)
end

function DI.pushforward(
    f::F,
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    dx_sametype = convert(typeof(x), only(tx))
    x_and_dx = Duplicated(x, dx_sametype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dy = only(autodiff(mode, f_and_df, x_and_dx, annotated_contexts...))
    return (dy,)
end

function DI.pushforward(
    f::F,
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    ty = only(autodiff(mode, f_and_df, x_and_tx, annotated_contexts...))
    return values(ty)
end

function DI.value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    # dy cannot be passed anyway
    y, new_ty = DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function DI.pushforward!(
    f::F,
    ty::NTuple,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    # dy cannot be passed anyway
    new_ty = DI.pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return ty
end

## Gradient

struct EnzymeForwardGradientPrep{B,O} <: DI.GradientPrep
    shadows::O
end

function EnzymeForwardGradientPrep(::Val{B}, shadows::O) where {B,O}
    return EnzymeForwardGradientPrep{B,O}(shadows)
end

function DI.prepare_gradient(
    f::F, backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}}, x
) where {F}
    valB = to_val(DI.pick_batchsize(backend, x))
    shadows = create_shadows(valB, x)
    return EnzymeForwardGradientPrep(valB, shadows)
end

function DI.gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    derivs = gradient(mode, f_and_df, x; chunk=Val(B), shadows=prep.shadows)
    return only(derivs)
end

function DI.value_and_gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    (; derivs, val) = gradient(mode, f_and_df, x; chunk=Val(B), shadows=prep.shadows)
    return val, only(derivs)
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    return copyto!(grad, DI.gradient(f, prep, backend, x))
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x)
    return y, copyto!(grad, new_grad)
end

## Jacobian

struct EnzymeForwardOneArgJacobianPrep{B,O} <: DI.JacobianPrep
    shadows::O
    output_length::Int
end

function EnzymeForwardOneArgJacobianPrep(
    ::Val{B}, shadows::O, output_length::Integer
) where {B,O}
    return EnzymeForwardOneArgJacobianPrep{B,O}(shadows, output_length)
end

function DI.prepare_jacobian(
    f::F, backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}}, x
) where {F}
    y = f(x)
    valB = to_val(DI.pick_batchsize(backend, x))
    shadows = create_shadows(valB, x)
    return EnzymeForwardOneArgJacobianPrep(valB, shadows, length(y))
end

function DI.jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F,B}
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    derivs = jacobian(mode, f_and_df, x; chunk=Val(B), shadows=prep.shadows)
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, prep.output_length, length(x))
end

function DI.value_and_jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F,B}
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    (; derivs, val) = jacobian(mode, f_and_df, x; chunk=Val(B), shadows=prep.shadows)
    jac_tensor = only(derivs)
    return val, maybe_reshape(jac_tensor, prep.output_length, length(x))
end

function DI.jacobian!(
    f::F,
    jac,
    prep::EnzymeForwardOneArgJacobianPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    return copyto!(jac, DI.jacobian(f, prep, backend, x))
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    prep::EnzymeForwardOneArgJacobianPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x)
    return y, copyto!(jac, new_jac)
end
