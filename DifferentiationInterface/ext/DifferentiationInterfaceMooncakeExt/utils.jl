get_config(::AnyAutoMooncake{Nothing}) = Config()
get_config(backend::AnyAutoMooncake{<:Config}) = backend.config

@inline zero_tangent_unwrap(c::DI.Context) = zero_tangent(DI.unwrap(c))
@inline first_unwrap(c, dc) = (DI.unwrap(c), dc)

function call_and_return(f!::F, y, x, contexts...) where {F}
    f!(y, x, contexts...)
    return y
end

function zero_tangent_or_primal(x, backend::AnyAutoMooncake)
    if get_config(backend).friendly_tangents
        # zero(x) but safer
        return tangent_to_primal!!(_copy_output(x), zero_tangent(x))
    else
        return zero_tangent(x)
    end
end

nanify(x::AbstractFloat) = convert(typeof(x), NaN)
nanify(x::AbstractArray) = map(nan_tangent, x)
nanify(x::Union{Tuple, NamedTuple}) = map(nan_tangent, x)
nanify(::NoFData) = NoFData()
nanify(::NoRData) = NoRData()

function nanify_fdata_and_rdata!!(contexts::Vararg{CoDual, C}) where {C}
    primal_contexts = map(primal, contexts)
    fdata_contexts = map(tangent, contexts)
    zero_rdata_contexts = map(zero_rdata, primal_contexts)
    foreach(fdata_contexts) do fc
        increment!!(fc, nanify(fc))
    end
    return map(nanify, zero_rdata_contexts)
end
