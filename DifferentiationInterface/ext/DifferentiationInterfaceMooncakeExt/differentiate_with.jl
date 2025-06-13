@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:Any}

struct MooncakeDifferentiateWithError <: Exception
    F::Type
    X::Type
    Y::Type
    function MooncakeDifferentiateWithError(::F, ::X, ::Y) where {F,X,Y}
        return new(F, X, Y)
    end
end

function Base.showerror(io::IO, e::MooncakeDifferentiateWithError)
    return print(
        io,
        "MooncakeDifferentiateWithError: For the function type $(e.F) and input type $(e.X), the output type $(e.Y) is currently not supported.",
    )
end

# For details, refer commented out test cases to see where the pullback creation fails.
function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:Number})
    primal_func = primal(dw)
    primal_x = primal(x)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (y.dx,))
        @assert rdata(only(tx)) isa rdata_type(tangent_type(typeof(primal_x)))
        return NoRData(), rdata(only(tx))
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert rdata(only(tx)) isa rdata_type(tangent_type(typeof(primal_x)))
        return NoRData(), rdata(only(tx))
    end

    pullback = if primal(y) isa Number
        pullback_scalar!!
    elseif primal(y) isa AbstractArray
        pullback_array!!
    else
        throw(MooncakeDifferentiateWithError(primal_func, primal_x, primal(y)))
    end

    return y, pullback
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:AbstractArray})
    primal_func = primal(dw)
    primal_x = primal(x)
    fdata_arg = x.dx
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (y.dx,))
        @assert rdata(first(only(tx))) isa rdata_type(tangent_type(typeof(first(primal_x))))
        fdata_arg .+= only(tx)
        return NoRData(), dy
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert rdata(first(only(tx))) isa rdata_type(tangent_type(typeof(first(primal_x))))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    pullback = if primal(y) isa Number
        pullback_scalar!!
    elseif primal(y) isa AbstractArray
        pullback_array!!
    else
        throw(MooncakeDifferentiateWithError(primal_func, primal_x, primal(y)))
    end

    return y, pullback
end
