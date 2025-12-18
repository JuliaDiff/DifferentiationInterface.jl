## Pushforward

struct EnzymeTwoArgPushforwardPrep{SIG, DF, DC} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    df!::DF
    context_shadows::DC
end

function DI.prepare_pushforward_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoEnzyme{<:Union{ForwardMode, Nothing}},
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C}
    ) where {F, B, C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    mode = forward_noprimal(backend)
    df! = function_shadow(f!, backend, mode, Val(B))
    context_shadows = make_context_shadows(backend, mode, Val(B), contexts...)
    return EnzymeTwoArgPushforwardPrep(_sig, df!, context_shadows)
end

function DI.value_and_pushforward(
        f!::F,
        y,
        prep::EnzymeTwoArgPushforwardPrep,
        backend::AutoEnzyme{<:Union{ForwardMode, Nothing}},
        x,
        tx::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; df!, context_shadows) = prep
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(1))
    dx = only(tx)
    dy = make_zero(y)
    x_and_dx = Duplicated(x, dx)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dy,)
end

function DI.value_and_pushforward(
        f!::F,
        y,
        prep::EnzymeTwoArgPushforwardPrep,
        backend::AutoEnzyme{<:Union{ForwardMode, Nothing}},
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; df!, context_shadows) = prep
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(B))
    ty = ntuple(_ -> make_zero(y), Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, ty
end

function DI.pushforward(
        f!::F,
        y,
        prep::EnzymeTwoArgPushforwardPrep,
        backend::AutoEnzyme{<:Union{ForwardMode, Nothing}},
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    _, ty = DI.value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
    return ty
end

function DI.value_and_pushforward!(
        f!::F,
        y,
        ty::NTuple{B},
        prep::EnzymeTwoArgPushforwardPrep,
        backend::AutoEnzyme{<:Union{ForwardMode, Nothing}},
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    (; df!, context_shadows) = prep
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, ty
end

function DI.pushforward!(
        f!::F,
        y,
        ty::NTuple,
        prep::EnzymeTwoArgPushforwardPrep,
        backend::AutoEnzyme{<:Union{ForwardMode, Nothing}},
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    return ty
end
