get_config(::AnyAutoMooncake{Nothing}) = Config()
get_config(backend::AnyAutoMooncake{<:Config}) = backend.config

@inline zero_tangent_unwrap(c::DI.Context) = zero_tangent(DI.unwrap(c))
@inline first_unwrap(c, dc) = (DI.unwrap(c), dc)

function call_and_return(f!::F, y, x, contexts...) where {F}
    f!(y, x, contexts...)
    return y
end

function adaptive_tangent_to_primal!!(primal, tangent)
    @static if new_friendly_tangents()
        # TODO: optimize performance by allocating cache during prep
        return Mooncake.tangent_to_friendly!!(primal, tangent)
    else
        return Mooncake.tangent_to_primal!!(primal, tangent)
    end
end

function zero_tangent_or_primal(x, backend::AnyAutoMooncake)
    if get_config(backend).friendly_tangents
        # zero(x) but safer
        return adaptive_tangent_to_primal!!(_copy_output(x), zero_tangent(x))
    else
        return zero_tangent(x)
    end
end
