## Pushforward

function DI.prepare_pushforward(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    return DI.NoPushforwardPrep(_sig)
end

function DI.value_and_pushforward(
    f::F,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    dx = only(tx)
    x_and_dx = Duplicated(x, dx)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dy, y = autodiff(mode, f_and_df, x_and_dx, annotated_contexts...)
    return y, (dy,)
end

function DI.value_and_pushforward(
    f::F,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode, Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    ty, y = autodiff(mode, f_and_df, x_and_tx, annotated_contexts...)
    return y, values(ty)
end

function DI.pushforward(
    f::F,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    dx = only(tx)
    x_and_dx = Duplicated(x, dx)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dy = only(autodiff(mode, f_and_df, x_and_dx, annotated_contexts...))
    return (dy,)
end

function DI.pushforward(
    f::F,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode, Val(B))
    x_and_tx = BatchDuplicated(x, tx)
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
    DI.check_prep(f, prep, backend, x, tx, contexts...)
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
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    # dy cannot be passed anyway
    new_ty = DI.pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return ty
end

## Gradient

struct EnzymeForwardGradientPrep{SIG,B,O} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    _valB::Val{B}
    shadows::O
end

function DI.prepare_gradient(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    valB = to_val(DI.pick_batchsize(backend, x))
    shadows = create_shadows(valB, x)
    return EnzymeForwardGradientPrep(_sig, valB, shadows)
end

function DI.gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{SIG,B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    derivs = gradient(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=prep.shadows
    )
    return first(derivs)
end

function DI.value_and_gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{SIG,B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    (; derivs, val) = gradient(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=prep.shadows
    )
    return val, first(derivs)
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeForwardGradientPrep{SIG,B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeForwardGradientPrep{SIG,B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

## Jacobian

struct EnzymeForwardOneArgJacobianPrep{SIG,B,O} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    _valB::Val{B}
    shadows::O
    output_length::Int
end

function DI.prepare_jacobian(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    y = f(x, map(DI.unwrap, contexts)...)
    valB = to_val(DI.pick_batchsize(backend, x))
    shadows = create_shadows(valB, x)
    return EnzymeForwardOneArgJacobianPrep(_sig, valB, shadows, length(y))
end

function DI.jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{SIG,B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    derivs = jacobian(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=prep.shadows
    )
    jac_tensor = first(derivs)
    return maybe_reshape(jac_tensor, prep.output_length, length(x))
end

function DI.value_and_jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{SIG,B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    (; derivs, val) = jacobian(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=prep.shadows
    )
    jac_tensor = first(derivs)
    return val, maybe_reshape(jac_tensor, prep.output_length, length(x))
end

function DI.jacobian!(
    f::F,
    jac,
    prep::EnzymeForwardOneArgJacobianPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return copyto!(jac, DI.jacobian(f, prep, backend, x, contexts...))
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    prep::EnzymeForwardOneArgJacobianPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x, contexts...)
    return y, copyto!(jac, new_jac)
end
