@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:AbstractArray}
@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:Number}

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:Number})
    primal_func = primal(dw)
    primal_x = primal(x)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (fdata(y.dx),))
        return NoRData(), only(tx)
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback!!(dy)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        return NoRData(), only(tx)
    end

    return y, pullback!!
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:AbstractArray})
    primal_func = primal(dw)
    primal_x = primal(x)
    fdata_arg = fdata(x.dx)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (fdata(y.dx),))
        fdata_arg .+= only(tx)
        return NoRData(), dy
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback!!(dy)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    # in case x is mutated when passed into f
    x = CoDual(primal_x, x.dx)
    return y, pullback!!
end
