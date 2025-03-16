## Pullback

struct EnzymeReverseTwoArgPullbackPrep{SIG,TY} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    ty_copy::TY
end

function DI.prepare_pullback(
    f!::F,
    y,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Val=Val(false),
) where {F,C}
    _sig = DI.signature(f!, y, backend, x, ty, contexts...; strict)
    ty_copy = map(copy, ty)
    return EnzymeReverseTwoArgPullbackPrep(_sig, ty_copy)
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    copyto!(only(prep.ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dy = only(prep.ty_copy)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dinputs = only(
        autodiff(mode, f!_and_df!, Const, y_and_dy, Active(x), annotated_contexts...)
    )
    dx = dinputs[2]
    return y, (dx,)
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    foreach(copyto!, prep.ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    ty = prep.ty_copy
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    dinputs = only(
        autodiff(mode, f!_and_df!, Const, y_and_ty, Active(x), annotated_contexts...)
    )
    tx = values(dinputs[2])
    return y, tx
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    copyto!(only(prep.ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dx = make_zero(x)  # allocates
    dy = only(prep.ty_copy)
    x_and_dx = Duplicated(x, dx)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dx,)
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    foreach(copyto!, prep.ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    tx = ntuple(_ -> make_zero(x), Val(B))  # allocates
    ty = prep.ty_copy
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, tx
end

function DI.value_and_pullback!(
    f!::F,
    y,
    tx::NTuple{1},
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    copyto!(only(prep.ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dx = only(tx)
    make_zero!(dx)
    dy = only(prep.ty_copy)
    x_and_dx = Duplicated(x, dx)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dx,)
end

function DI.value_and_pullback!(
    f!::F,
    y,
    tx::NTuple{B},
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    foreach(copyto!, prep.ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    make_zero!(tx)
    ty = prep.ty_copy
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, tx
end
