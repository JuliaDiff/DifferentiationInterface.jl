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
        # Mooncake 0.5.25+ replaced `tangent_to_primal!!` with the
        # `tangent_to_friendly!!` framework. For this internal backup we still
        # need a primal-shaped value, so use the `AsPrimal` path when
        # available and fall back for older Mooncake releases.
        return tangent_to_user_primal(zero_tangent(x), x)
    else
        return zero_tangent(x)
    end
end

@inline maybe_getfield(mod, name::Symbol) =
    isdefined(mod, name) ? getfield(mod, name) : nothing

const mooncake_tangent_to_friendly = maybe_getfield(
    Mooncake, Symbol("tangent_to_friendly!!")
)
const mooncake_friendly_tangent_cache = maybe_getfield(Mooncake, :FriendlyTangentCache)
const mooncake_as_primal = maybe_getfield(Mooncake, :AsPrimal)
const mooncake_no_cache = maybe_getfield(Mooncake, :NoCache)

function tangent_to_user_primal(tx, x)
    if !isnothing(mooncake_tangent_to_friendly) &&
            !isnothing(mooncake_friendly_tangent_cache) &&
            !isnothing(mooncake_as_primal) &&
            !isnothing(mooncake_no_cache)
        dest = mooncake_friendly_tangent_cache{mooncake_as_primal}(_copy_output(x))
        cache = isbitstype(typeof(x)) ? mooncake_no_cache() : IdDict{Any, Any}()
        return mooncake_tangent_to_friendly(dest, x, tx, cache)
    else
        return tangent_to_primal!!(_copy_output(x), tx)
    end
end
