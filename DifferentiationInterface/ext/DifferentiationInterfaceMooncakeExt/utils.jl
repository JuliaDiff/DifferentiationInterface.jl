get_config(::AnyAutoMooncake{Nothing}) = Config()
get_config(backend::AnyAutoMooncake{<:Config}) = backend.config

@inline zero_tangent_unwrap(c::DI.Context) = zero_tangent(DI.unwrap(c))
@inline first_unwrap(c, dc) = (DI.unwrap(c), dc)

function call_and_return(f!::F, y, x, contexts...) where {F}
    f!(y, x, contexts...)
    return y
end

# Hook for bridging primal types whose `friendly_tangent_cache` falls through to
# `:as_raw` in Mooncake's framework, leaking a raw `Tangent` / `MutableTangent`
# instead of a primal-shaped value. The default returns `nothing`; specialised
# methods are loaded by triple-extensions when the relevant primal-type packages
# are available (see DifferentiationInterfaceMooncakeStaticArraysExt for the
# `SArray` / `MArray` case).
_to_friendly_value(t, x) = nothing

function zero_tangent_or_primal(x, backend::AnyAutoMooncake)
    zt = zero_tangent(x)
    if get_config(backend).friendly_tangents
        return @something(_to_friendly_value(zt, x), tangent_to_friendly!!(x, zt))
    else
        return zt
    end
end
