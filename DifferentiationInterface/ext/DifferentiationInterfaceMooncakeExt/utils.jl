const AutoMooncakeForwardOverReverse{C1, C2} = DI.SecondOrder{
    AutoMooncakeForward{C1},
    AutoMooncake{C2},
}

get_config(::AnyAutoMooncake{Nothing}) = Config()
get_config(backend::AnyAutoMooncake{<:Config}) = backend.config
get_config(backend::AutoMooncakeForwardOverReverse) = get_config(DI.outer(backend)) # TODO: mix?

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

# a copy of `x` with the same type and all its differentiable data set to zero
# (`FriendlyTangentCache` and `AsPrimal` exist since Mooncake v0.5.25, the compat lower bound)
function zero_primal(x)
    cache = Mooncake.FriendlyTangentCache{Mooncake.AsPrimal}(_copy_output(x))
    return Mooncake.tangent_to_friendly!!(cache, x, zero_tangent(x), IdDict{Any, Any}())
end

"""
    input_tangent(x, dx, backend)

Convert the tangent `dx` provided by DI for an input `x` into what Mooncake expects.

DI builds tangents of an array `x` with `similar(x)`, which is an `Array` for many other array types (`SubArray`, `Transpose`, ...).
Mooncake expects a tangent of type `tangent_type(typeof(x))`, or with friendly tangents an array of the same type as `x`.
"""
input_tangent(x, dx, ::AnyAutoMooncake) = dx

function input_tangent(x::AbstractArray, dx::AbstractArray, backend::AnyAutoMooncake)
    friendly = get_config(backend).friendly_tangents
    if dx isa tangent_type(typeof(x)) || (friendly && dx isa typeof(x))
        return dx
    end
    dx_like_x = copyto!(zero_primal(x), dx)
    if friendly
        return dx_like_x
    else
        return Mooncake.primal_to_tangent!!(zero_tangent(x), dx_like_x)
    end
end
