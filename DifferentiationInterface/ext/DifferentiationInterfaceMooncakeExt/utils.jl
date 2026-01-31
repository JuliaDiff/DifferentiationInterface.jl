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
