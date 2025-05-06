# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
function DI.pick_batchsize(::AutoEnzyme, N::Integer)
    B = DI.reasonable_batchsize(N, 16)
    return DI.BatchSizeSettings{B}(N)
end

to_val(::DI.BatchSizeSettings{B}) where {B} = Val(B)

## Annotations

const AnyDuplicated = Union{
    Duplicated,
    MixedDuplicated,
    BatchDuplicated,
    BatchMixedDuplicated,
    DuplicatedNoNeed,
    BatchDuplicatedNoNeed,
}

function get_f_and_df(f::F, ::AutoEnzyme{M,Nothing}, ::Val{B}) where {F,M,B}
    return f
end

function get_f_and_df(f::F, ::AutoEnzyme{M,<:Const}, ::Val{B}) where {F,M,B}
    return Const(f)
end

function get_f_and_df(f::F, backend::AutoEnzyme{M,<:AnyDuplicated}, ::Val{B}) where {F,M,B}
    # TODO: needs more sophistication for mixed activities
    df = function_shadow(f, backend, Val(B))
    if B == 1
        return Duplicated(f, df)
    else
        return BatchDuplicated(f, df)
    end
end

function function_shadow(
    ::F, ::AutoEnzyme{M,<:Union{Const,Nothing}}, ::Val{B}
) where {M,B,F}
    return nothing
end

function function_shadow(f::F, ::AutoEnzyme{M,<:AnyDuplicated}, ::Val{B}) where {F,M,B}
    if B == 1
        return make_zero(f)
    else
        return ntuple(_ -> make_zero(f), Val(B))
    end
end

function get_f_and_df_prepared!(_df, f::F, ::AutoEnzyme{M,Nothing}, ::Val{B}) where {F,M,B}
    return f
end

function get_f_and_df_prepared!(_df, f::F, ::AutoEnzyme{M,<:Const}, ::Val{B}) where {F,M,B}
    return Const(f)
end

function get_f_and_df_prepared!(
    df, f::F, ::AutoEnzyme{M,<:AnyDuplicated}, ::Val{B}
) where {F,M,B}
    make_zero!(df)
    if B == 1
        return Duplicated(f, df)
    else
        return BatchDuplicated(f, df)
    end
end

force_annotation(f::F) where {F<:Annotation} = f
force_annotation(f::F) where {F} = Const(f)

function _translate(::AutoEnzyme, ::Mode, ::Val{B}, c_wrapped::DI.Constant) where {B}
    c = DI.unwrap(c_wrapped)
    return Const(c)
end

function _translate(::AutoEnzyme, ::Mode, ::Val{B}, c_wrapped::DI.Cache) where {B}
    c = DI.unwrap(c_wrapped)
    if B == 1
        dc = make_zero(c)
        return Duplicated(c, dc)
    else
        dc = ntuple(_ -> make_zero(c), Val(B))
        return BatchDuplicated(c, dc)
    end
end

function _translate(
    backend::AutoEnzyme, mode::Mode, ::Val{B}, c_wrapped::DI.ConstantOrCache
) where {B}
    c = DI.unwrap(c_wrapped)
    IA = guess_activity(typeof(c), mode)
    if IA <: Const
        return _translate(backend, mode, Val(B), DI.Constant(c))
    else
        return _translate(backend, mode, Val(B), DI.Cache(c))
    end
end

function _translate(
    backend::AutoEnzyme, ::Mode, ::Val{B}, c_wrapped::DI.FunctionContext
) where {B}
    f = DI.unwrap(c_wrapped)
    return force_annotation(get_f_and_df(f, backend, Val(B)))
end

function translate(
    backend::AutoEnzyme, mode::Mode, ::Val{B}, contexts::Vararg{DI.Context,C}
) where {B,C}
    new_contexts = map(contexts) do c_wrapped
        _translate(backend, mode, Val(B), c_wrapped)
    end
    return new_contexts
end

function _shadow(::AutoEnzyme, ::Mode, ::Val{B}, c_wrapped::DI.Constant) where {B}
    return nothing
end

function _shadow(::AutoEnzyme, ::Mode, ::Val{B}, c_wrapped::DI.Cache) where {B}
    c = DI.unwrap(c_wrapped)
    if B == 1
        return make_zero(c)
    else
        return ntuple(_ -> make_zero(c), Val(B))
    end
end

function _shadow(
    backend::AutoEnzyme, mode::Mode, valB::Val{B}, c_wrapped::DI.ConstantOrCache
) where {B}
    c = DI.unwrap(c_wrapped)
    IA = guess_activity(typeof(c), mode)
    if IA <: Const
        return _shadow(backend, mode, valB, DI.Constant(c))
    else
        return _shadow(backend, mode, valB, DI.Cache(c))
    end
end

function _shadow(
    backend::AutoEnzyme{M,<:Union{Const,Nothing}},
    ::Mode,
    ::Val{B},
    c_wrapped::DI.FunctionContext,
) where {M,B}
    f = DI.unwrap(c_wrapped)
    return function_shadow(f, backend, Val(B))
end

function shadows(
    backend::AutoEnzyme, mode::Mode, ::Val{B}, contexts::Vararg{DI.Context,C}
) where {B,C}
    context_shadows = map(contexts) do c_wrapped
        _shadow(backend, mode, Val(B), c_wrapped)
    end
    return context_shadows
end

function _translate_prepared!(dc, c_wrapped::DI.Constant, ::Val{B}) where {B}
    c = DI.unwrap(c_wrapped)
    return Const(c)
end

function _translate_prepared!(dc, c_wrapped::DI.Cache, ::Val{B}) where {B}
    c = DI.unwrap(c_wrapped)
    make_zero!(dc)
    if B == 1
        return Duplicated(c, dc)
    else
        return BatchDuplicated(c, dc)
    end
end

function _translate_prepared!(
    dc, c_wrapped::Union{DI.ConstantOrCache,DI.FunctionContext}, ::Val{B}
) where {B}
    c = DI.unwrap(c_wrapped)
    if isnothing(dc)
        return Const(c)
    else
        # make_zero!(dc)  # doesn't work because of immutable values
        if B == 1
            dc_new = make_zero(c)
            return Duplicated(c, dc_new)
        else
            dc_new = ntuple(_ -> make_zero(c), Val(B))
            return BatchDuplicated(c, dc_new)
        end
    end
end

function translate_prepared!(
    context_shadows::NTuple{C,Any}, contexts::NTuple{C,DI.Context}, ::Val{B}
) where {B,C}
    new_contexts = map(context_shadows, contexts) do dc, c_wrapped
        _translate_prepared!(dc, c_wrapped, Val(B))
    end
    return new_contexts
end

## Modes

forward_noprimal(backend::AutoEnzyme{<:ForwardMode}) = NoPrimal(backend.mode)
forward_noprimal(::AutoEnzyme{Nothing}) = Forward

forward_withprimal(backend::AutoEnzyme{<:ForwardMode}) = WithPrimal(backend.mode)
forward_withprimal(::AutoEnzyme{Nothing}) = ForwardWithPrimal

reverse_noprimal(backend::AutoEnzyme{<:ReverseMode}) = NoPrimal(backend.mode)
reverse_noprimal(::AutoEnzyme{Nothing}) = Reverse

reverse_withprimal(backend::AutoEnzyme{<:ReverseMode}) = WithPrimal(backend.mode)
reverse_withprimal(::AutoEnzyme{Nothing}) = ReverseWithPrimal

function reverse_split_withprimal(backend::AutoEnzyme{<:ReverseMode})
    return set_err(WithPrimal(Split(backend.mode)), backend)
end

function reverse_split_withprimal(backend::AutoEnzyme{Nothing})
    return set_err(ReverseSplitWithPrimal, backend)
end

function set_err(mode::Mode, backend::AutoEnzyme{<:Any,Nothing})
    return EnzymeCore.set_err_if_func_written(mode)
end
set_err(mode::Mode, backend::AutoEnzyme{<:Any,<:Annotation}) = mode

function maybe_reshape(A::AbstractMatrix, m, n)
    @assert size(A) == (m, n)
    return A
end

function maybe_reshape(A::AbstractArray, m, n)
    return reshape(A, m, n)
end

annotate(::Type{Active{T}}, x, dx) where {T} = Active(x)
annotate(::Type{Duplicated{T}}, x, dx) where {T} = Duplicated(x, dx)

function annotate(::Type{BatchDuplicated{T,B}}, x, tx::NTuple{B}) where {T,B}
    return BatchDuplicated(x, tx)
end

batchify_activity(::Type{Active{T}}, ::Val{B}) where {T,B} = Active{T}
batchify_activity(::Type{Duplicated{T}}, ::Val{B}) where {T,B} = BatchDuplicated{T,B}
