## Pushforward

function DI.prepare_pushforward(
    f!::F,
    y,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {F,C}
    SIG = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    return DI.NoPushforwardPrep{SIG}()
end

function DI.value_and_pushforward(
    f!::F,
    y,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dx = only(tx)
    dy = make_zero(y)
    x_and_dx = Duplicated(x, dx)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dy,)
end

function DI.value_and_pushforward(
    f!::F,
    y,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    ty = ntuple(_ -> make_zero(y), Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, ty
end

function DI.pushforward(
    f!::F,
    y,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    _, ty = DI.value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
    return ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple{B},
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    DI.value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    return ty
end
