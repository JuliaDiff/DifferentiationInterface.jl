function seeded_autodiff_thunk(
    rmode::ReverseModeSplit{ReturnPrimal},
    dresult,
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,FA<:Annotation,RA<:Annotation,N}
    forward, reverse = autodiff_thunk(rmode, FA, RA, typeof.(args)...)
    tape, result, shadow_result = forward(f, args...)
    if RA <: Active
        dinputs = only(reverse(f, args..., dresult, tape))
    else
        shadow_result .+= dresult  # TODO: generalize beyond arrays
        dinputs = only(reverse(f, args..., tape))
    end
    if ReturnPrimal
        return (dinputs, result)
    else
        return (dinputs,)
    end
end

function batch_seeded_autodiff_thunk(
    rmode::ReverseModeSplit{ReturnPrimal},
    dresults::NTuple{B},
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,B,FA<:Annotation,RA<:Annotation,N}
    rmode_rightwidth = ReverseSplitWidth(rmode, Val(B))
    forward, reverse = autodiff_thunk(rmode_rightwidth, FA, RA, typeof.(args)...)
    tape, result, shadow_results = forward(f, args...)
    if RA <: Active
        dinputs = only(reverse(f, args..., dresults, tape))
    else
        foreach(shadow_results, dresults) do d0, d
            d0 .+= d  # use recursive_add here?
        end
        dinputs = only(reverse(f, args..., tape))
    end
    if ReturnPrimal
        return (dinputs, result)
    else
        return (dinputs,)
    end
end

## Pullback

struct EnzymeReverseOneArgPullbackPrep{SIG,DF,DC,Y} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    df::DF
    context_shadows::DC
    y_example::Y  # useful to create return activity
end

function DI.prepare_pullback_nokwarg(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C};
) where {F,B,C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    df = function_shadow(f, backend, Val(B))
    mode = reverse_split_withprimal(backend)
    context_shadows = make_context_shadows(backend, mode, Val(B), contexts...)
    y = f(x, map(DI.unwrap, contexts)...)
    return EnzymeReverseOneArgPullbackPrep(_sig, df, context_shadows, y)
end

### Out-of-place

function DI.value_and_pullback(
    f::F,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; df, context_shadows, y_example) = prep
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df_prepared!(df, f, backend, Val(1)))
    IA = guess_activity(typeof(x), mode)
    RA = guess_activity(typeof(y_example), mode)
    dx = make_zero(x)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    dinputs, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, annotate(IA, x, dx), annotated_contexts...
    )
    new_dx = first(dinputs)
    if isnothing(new_dx)
        return result, (dx,)
    else
        return result, (new_dx,)
    end
end

function DI.value_and_pullback(
    f::F,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; df, context_shadows, y_example) = prep
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df_prepared!(df, f, backend, Val(B)))
    IA = batchify_activity(guess_activity(typeof(x), mode), Val(B))
    RA = batchify_activity(guess_activity(typeof(y_example), mode), Val(B))
    tx = ntuple(_ -> make_zero(x), Val(B))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    dinputs, result = batch_seeded_autodiff_thunk(
        mode, ty, f_and_df, RA, annotate(IA, x, tx), annotated_contexts...
    )
    new_tx = values(first(dinputs))
    if isnothing(new_tx)
        return result, tx
    else
        return result, new_tx
    end
end

function DI.pullback(
    f::F,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return last(DI.value_and_pullback(f, prep, backend, x, ty, contexts...))
end

### In-place

function DI.value_and_pullback!(
    f::F,
    tx::NTuple{1},
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; df, context_shadows, y_example) = prep
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df_prepared!(df, f, backend, Val(1)))
    RA = guess_activity(typeof(y_example), mode)
    dx = only(tx)
    make_zero!(dx)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    _, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Duplicated(x, dx), annotated_contexts...
    )
    return result, tx
end

function DI.value_and_pullback!(
    f::F,
    tx::NTuple{B},
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; df, context_shadows, y_example) = prep
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df_prepared!(df, f, backend, Val(B)))
    RA = batchify_activity(guess_activity(typeof(y_example), mode), Val(B))
    make_zero!(tx)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(B))
    _, result = batch_seeded_autodiff_thunk(
        mode, ty, f_and_df, RA, BatchDuplicated(x, tx), annotated_contexts...
    )
    return result, tx
end

function DI.pullback!(
    f::F,
    tx::NTuple,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return last(DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...))
end

## Gradient

struct EnzymeGradientPrep{SIG,DF,DC} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    df::DF
    context_shadows::DC
end

function DI.prepare_gradient_nokwarg(
    strict::Val,
    f::F,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    df = function_shadow(f, backend, Val(1))
    mode = reverse_withprimal(backend)
    context_shadows = make_context_shadows(backend, mode, Val(1), contexts...)
    return EnzymeGradientPrep(_sig, df, context_shadows)
end

### Enzyme gradient API (only constants)

function DI.gradient(
    f::F,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows) = prep
    mode = reverse_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    grads = gradient(mode, f_and_df, x, annotated_contexts...)
    return first(grads)
end

function DI.value_and_gradient(
    f::F,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{DI.Constant,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows) = prep
    mode = reverse_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    grads, result = gradient(mode, f_and_df, x, annotated_contexts...)
    return result, first(grads)
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    DI.check_prep(f, prep, backend, x)
    (; df) = prep
    mode = reverse_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    gradient!(mode, grad, f_and_df, x)
    return grad
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    DI.check_prep(f, prep, backend, x)
    (; df) = prep
    mode = reverse_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    _, result = gradient!(mode, grad, f_and_df, x)
    return result, grad
end

### Generic

function DI.gradient(
    f::F,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows) = prep
    mode = reverse_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    IA = guess_activity(typeof(x), mode)
    grad = make_zero(x)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    dinputs = only(
        autodiff(mode, f_and_df, Active, annotate(IA, x, grad), annotated_contexts...)
    )
    new_grad = first(dinputs)
    if isnothing(new_grad)
        return grad
    else
        return new_grad
    end
end

function DI.value_and_gradient(
    f::F,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows) = prep
    mode = reverse_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    IA = guess_activity(typeof(x), mode)
    grad = make_zero(x)
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    dinputs, result = autodiff(
        mode, f_and_df, Active, annotate(IA, x, grad), annotated_contexts...
    )
    new_grad = first(dinputs)
    if isnothing(new_grad)
        return result, grad
    else
        return result, new_grad
    end
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows) = prep
    mode = reverse_noprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    make_zero!(grad)
    autodiff(mode, f_and_df, Active, Duplicated(x, grad), annotated_contexts...)
    return grad
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.check_prep(f, prep, backend, x, contexts...)
    (; df, context_shadows) = prep
    mode = reverse_withprimal(backend)
    f_and_df = get_f_and_df_prepared!(df, f, backend, Val(1))
    annotated_contexts = translate_prepared!(context_shadows, contexts, Val(1))
    make_zero!(grad)
    _, y = autodiff(mode, f_and_df, Active, Duplicated(x, grad), annotated_contexts...)
    return y, grad
end
