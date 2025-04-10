@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:AbstractArray}
@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:Number}

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:Number})
    primal_func = primal(dw)
    primal_x = primal(x)
    (; f, backend) = primal_func
    y = f(primal_x)

    function pullback!!(dy)
        tx = DI.pullback(f, backend, primal_x, (dy...,))
        return NoRData(), only(tx)
    end

    return zero_fcodual(y), pullback!!
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:AbstractArray})
    primal_func = primal(dw)
    primal_x = primal(x)
    fdata_arg = fdata(x.dx)
    (; f, backend) = primal_func
    y = f(primal_x)

    function pullback!!(dy)
        tx = DI.pullback(f, backend, primal_x, (dy...,))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    return zero_fcodual(y), pullback!!
end
