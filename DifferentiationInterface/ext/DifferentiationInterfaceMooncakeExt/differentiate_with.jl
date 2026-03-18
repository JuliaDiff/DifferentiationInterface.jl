const NumberOrArray = Union{Number, AbstractArray{<:Number}}

# Mark DifferentiateWith with a range of context arities as primitives.
# For C contexts, the corresponding call tuple type is
# Tuple{DI.DifferentiateWith{C}, Any, Vararg{Any, C}}:
# one slot for the primal input x and C slots for contexts.
for C in 0:16
    @eval @is_primitive MinimalCtx Tuple{DI.DifferentiateWith{$C}, Vararg{Any, $(C + 1)}}
end
struct MooncakeDifferentiateWithError <: Exception
    F::Type
    X::Type
    Y::Type
    function MooncakeDifferentiateWithError(::F, ::X, ::Y) where {F, X, Y}
        return new(F, X, Y)
    end
end

function Base.showerror(io::IO, e::MooncakeDifferentiateWithError)
    return print(
        io,
        "MooncakeDifferentiateWithError: For the function type `$(e.F)` and input types `$(e.X)`, the output type `$(e.Y)` is currently not supported.",
    )
end

function Mooncake.rrule!!(
        dw::CoDual{<:DI.DifferentiateWith{C}},
        x::CoDual{<:Number},
        contexts::Vararg{CoDual{<:NumberOrArray}, C}
    ) where {C}
    @assert tangent_type(typeof(dw)) == NoTangent
    primal_func = primal(dw)
    primal_x = primal(x)
    primal_contexts = map(primal, contexts)
    (; f, backend, context_wrappers) = primal_func
    y = zero_fcodual(f(primal_x, primal_contexts...))
    wrapped_primal_contexts = map(DI.call, context_wrappers, primal_contexts)

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        dx = DI.pullback(f, backend, primal_x, (y.dx,), wrapped_primal_contexts...) |> only
        @assert rdata(dx) isa rdata_type(tangent_type(typeof(primal_x)))
        rc = nanify_fdata_and_rdata!!(contexts...)
        return (NoRData(), rdata(dx), rc...)
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        dx = DI.pullback(f, backend, primal_x, (dy,), wrapped_primal_contexts...) |> only
        @assert rdata(dx) isa rdata_type(tangent_type(typeof(primal_x)))
        rc = nanify_fdata_and_rdata!!(contexts...)
        return (NoRData(), rdata(dx), rc...)
    end

    pullback = if primal(y) isa Number
        pullback_scalar!!
    elseif primal(y) isa AbstractArray
        pullback_array!!
    else
        throw(MooncakeDifferentiateWithError(primal_func, (primal_x, primal_contexts...), primal(y)))
    end

    return y, pullback
end

function Mooncake.rrule!!(
        dw::CoDual{<:DI.DifferentiateWith{C}},
        x::CoDual{<:AbstractArray{<:Number}},
        contexts::Vararg{<:CoDual, C}
    ) where {C}
    @assert tangent_type(typeof(dw)) == NoTangent
    primal_func = primal(dw)
    primal_x = primal(x)
    primal_contexts = map(primal, contexts)
    (; f, backend, context_wrappers) = primal_func
    y = zero_fcodual(f(primal_x, primal_contexts...))
    wrapped_primal_contexts = map(DI.call, context_wrappers, primal_contexts)

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        dx = DI.pullback(f, backend, primal_x, (y.dx,), wrapped_primal_contexts...) |> only
        @assert rdata(dx) isa rdata_type(tangent_type(typeof(primal_x)))
        x.dx .+= dx
        rc = nanify_fdata_and_rdata!!(contexts...)
        return (NoRData(), dy, rc...)
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        dx = DI.pullback(f, backend, primal_x, (dy,), wrapped_primal_contexts...) |> only
        @assert rdata(dx) isa rdata_type(tangent_type(typeof(primal_x)))
        x.dx .+= dx
        rc = nanify_fdata_and_rdata!!(contexts...)
        return (NoRData(), NoRData(), rc...)
    end

    pullback = if primal(y) isa Number
        pullback_scalar!!
    elseif primal(y) isa AbstractArray
        pullback_array!!
    else
        throw(MooncakeDifferentiateWithError(primal_func, (primal_x, primal_contexts...), primal(y)))
    end

    return y, pullback
end
