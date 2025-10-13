## Pullback

struct EnzymeReverseTwoArgPullbackPrep{SIG, DF, DC, TY} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    df!::DF
    context_shadows::DC
    ty_copy::TY
end

function DI.prepare_pullback_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x,
        ty::NTuple{B},
        contexts::Vararg{DI.Context, C}
    ) where {F, B, C}
    _sig = DI.signature(f!, y, backend, x, ty, contexts...; strict)
    df! = function_shadow(f!, backend, Val(B))
    mode = reverse_noprimal(backend)
    context_shadows = make_context_shadows(backend, mode, Val(B), contexts...)
    ty_copy = map(copy, ty)
    return EnzymeReverseTwoArgPullbackPrep(_sig, df!, context_shadows, ty_copy)
end

function DI.value_and_pullback(
        f!::F,
        y,
        prep::EnzymeReverseTwoArgPullbackPrep,
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x::Number,
        ty::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; df!, context_shadows, ty_copy) = prep
    copyto!(only(ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(1))
    dy = only(ty_copy)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
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
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x::Number,
        ty::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; df!, context_shadows, ty_copy) = prep
    foreach(copyto!, ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(B))
    ty = ty_copy
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
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
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x,
        ty::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; df!, context_shadows, ty_copy) = prep
    copyto!(only(ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(1))
    dx = make_zero(x)  # allocates
    dy = only(ty_copy)
    x_and_dx = Duplicated(x, dx)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dx,)
end

function DI.value_and_pullback(
        f!::F,
        y,
        prep::EnzymeReverseTwoArgPullbackPrep,
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x,
        ty::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; df!, context_shadows, ty_copy) = prep
    foreach(copyto!, ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(B))
    tx = ntuple(_ -> make_zero(x), Val(B))  # allocates
    ty = ty_copy
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, tx
end

function DI.value_and_pullback!(
        f!::F,
        y,
        tx::NTuple{1},
        prep::EnzymeReverseTwoArgPullbackPrep,
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x,
        ty::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; df!, context_shadows, ty_copy) = prep
    copyto!(only(ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(1))
    dx = only(tx)
    make_zero!(dx)
    dy = only(ty_copy)
    x_and_dx = Duplicated(x, dx)
    y_and_dy = Duplicated(y, dy)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dx,)
end

function DI.value_and_pullback!(
        f!::F,
        y,
        tx::NTuple{B},
        prep::EnzymeReverseTwoArgPullbackPrep,
        backend::AutoEnzyme{<:Union{ReverseMode, Nothing}},
        x,
        ty::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    (; df!, context_shadows, ty_copy) = prep
    foreach(copyto!, ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df_prepared!(df!, f!, backend, Val(B))
    make_zero!(tx)
    ty = ty_copy
    x_and_tx = BatchDuplicated(x, tx)
    y_and_ty = BatchDuplicated(y, ty)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, tx
end
