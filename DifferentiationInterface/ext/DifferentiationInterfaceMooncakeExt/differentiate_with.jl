@is_primitive MinimalCtx Tuple{CoDual{<:DI.DifferentiateWith},CoDual{<:AbstractArray}}
@is_primitive MinimalCtx Tuple{CoDual{<:DI.DifferentiateWith},CoDual{<:Number}}

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, args::CoDual...)
    primal_func = Mooncake.primal(dw)
    primal_args = map(arg -> Mooncake.primal(arg), args)

    (; f, backend) = primal_func
    y = f(primal_args...)

    prep_same = DI.prepare_pullback_same_point_nokwarg(
        Val(true), f, backend, primal_args..., (y,)
    )

    function pullback!!(dy)
        tx = DI.pullback(f, prep_same, backend, primal_args, (dy,))
        args_rdata = map((x) -> (x, Mooncake.zero_rdata(x)), only(tx))
        return NoRData(), args_rdata...
    end

    return zero_fcodual(y), pullback!!
end
