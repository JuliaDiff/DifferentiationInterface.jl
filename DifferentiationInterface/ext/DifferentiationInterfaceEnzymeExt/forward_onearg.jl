## Pushforward

struct EnzymeOneArgPushforwardPrep{SIG,DF,DC} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    df::DF
    context_shadows::DC
end

function DI.prepare_pushforward_nokwarg(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C};
) where {F,C,B}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    df = function_shadow(f, backend, Val(B))
    mode = forward_withprimal(backend)
    context_shadows = make_context_shadows(backend, mode, Val(B), contexts...)
    return EnzymeOneArgPushforwardPrep(_sig, df, context_shadows)
end

function DI.value_and_pushforward(
    f::F,
    prep::EnzymeOneArgPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; df, context_shadows) = prep
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    dx = only(tx)
    x_and_dx = Duplicated(x, dx)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    dy, y = autodiff(mode, f_and_df, x_and_dx, annotated_contexts...)
    dy = runtime_activity_safeguard(backend, y, dy)
    return y, (dy,)
end

function DI.value_and_pushforward(
    f::F,
    prep::EnzymeOneArgPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; df, context_shadows) = prep
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    ty_nt, y = autodiff(mode, f_and_df, x_and_tx, annotated_contexts...)
    ty = values(ty_nt)
    ty = runtime_activity_safeguard(backend, y, ty)
    return y, ty
end

function DI.pushforward(
    f::F,
    prep::EnzymeOneArgPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    if has_runtime_activity(backend)
        return DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)[2]
    end
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; df, context_shadows) = prep
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    dx = only(tx)
    x_and_dx = Duplicated(x, dx)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    dy = only(autodiff(mode, f_and_df, x_and_dx, annotated_contexts...))
    return (dy,)
end

function DI.pushforward(
    f::F,
    prep::EnzymeOneArgPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    if has_runtime_activity(backend)
        return DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)[2]
    end
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    (; df, context_shadows) = prep
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    ty_nt = only(autodiff(mode, f_and_df, x_and_tx, annotated_contexts...))
    ty = values(ty_nt)
    return ty
end

function DI.value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::EnzymeOneArgPushforwardPrep,
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
    prep::EnzymeOneArgPushforwardPrep,
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

struct EnzymeForwardGradientPrep{SIG,B,DF,DC,O} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    _valB::Val{B}
    df::DF
    context_shadows::DC
    basis_shadows::O
end

function DI.prepare_gradient_nokwarg(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    valB = to_val(DI.pick_batchsize(backend, x))
    df = function_shadow(f, backend, valB)
    mode = forward_withprimal(backend)
    context_shadows = make_context_shadows(backend, mode, valB, contexts...)
    basis_shadows = create_shadows(valB, x)
    return EnzymeForwardGradientPrep(_sig, valB, df, context_shadows, basis_shadows)
end

function DI.gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{SIG,B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows, basis_shadows) = prep
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(B))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    derivs = gradient(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=basis_shadows
    )
    deriv = first(derivs)
    deriv = runtime_activity_safeguard(backend, x, deriv)
    return deriv
end

function DI.value_and_gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{SIG,B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows, basis_shadows) = prep
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(B))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    (; derivs, val) = gradient(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=basis_shadows
    )
    deriv = first(derivs)
    deriv = runtime_activity_safeguard(backend, x, deriv)
    return val, deriv
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

struct EnzymeForwardOneArgJacobianPrep{SIG,B,DF,DC,O} <: DI.JacobianPrep{SIG}
    _sig::Val{SIG}
    _valB::Val{B}
    df::DF
    context_shadows::DC
    basis_shadows::O
    output_length::Int
end

function DI.prepare_jacobian_nokwarg(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    y = f(x, map(DI.unwrap, contexts)...)
    valB = to_val(DI.pick_batchsize(backend, x))
    mode = forward_withprimal(backend)
    df = function_shadow(f, backend, valB)
    context_shadows = make_context_shadows(backend, mode, valB, contexts...)
    basis_shadows = create_shadows(valB, x)
    return EnzymeForwardOneArgJacobianPrep(
        _sig, valB, df, context_shadows, basis_shadows, length(y)
    )
end

function DI.jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{SIG,B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows, basis_shadows, output_length) = prep
    mode = forward_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(B))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    derivs = jacobian(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=basis_shadows
    )
    jac_tensor = first(derivs)
    return maybe_reshape(jac_tensor, output_length, length(x))
end

function DI.value_and_jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{SIG,B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,SIG,B,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows, basis_shadows, output_length) = prep
    mode = forward_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(B))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    (; derivs, val) = jacobian(
        mode, f_and_df, x, annotated_contexts...; chunk=Val(B), shadows=basis_shadows
    )
    jac_tensor = first(derivs)
    return val, maybe_reshape(jac_tensor, output_length, length(x))
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
