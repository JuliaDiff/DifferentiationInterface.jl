@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:Union{Number,AbstractArray}}

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:Number})
    primal_func = primal(dw)
    primal_x = primal(x)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (fdata(y.dx),))
        @assert only(tx) isa rdata_type(typeof(primal_x))
        return NoRData(), only(tx)
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert only(tx) isa rdata_type(typeof(primal_x))
        return NoRData(), only(tx)
    end

    return y, typeof(primal(y)) <: Number ? pullback_scalar!! : pullback_array!!
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:AbstractArray})
    primal_func = primal(dw)
    primal_x = primal(x)
    fdata_arg = fdata(x.dx)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (fdata(y.dx),))
        fdata_arg .+= only(tx)
        return NoRData(), dy
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    return y, typeof(primal(y)) <: Number ? pullback_scalar!! : pullback_array!!
end
