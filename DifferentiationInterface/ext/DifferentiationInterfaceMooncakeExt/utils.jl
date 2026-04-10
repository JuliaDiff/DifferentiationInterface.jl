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
        # zero(x) but safer
        return tangent_to_primal!!(_copy_output(x), zero_tangent(x))
    else
        return zero_tangent(x)
    end
end

# When the primal is a struct-backed array (e.g. `ComponentArray`, `MVector`)
# or a struct whose `tangent_type` is `Tangent` / `MutableTangent`,
# `value_and_gradient!!` and friends return the differential as the tangent
# wrapper rather than something whose layout matches the primal.  Downstream
# code (`copyto!`, iteration, OptimizationBase, `≈` against the expected
# primal-shaped result) expects a value with the same shape as the primal,
# so we unwrap here.
#
# `tangent_to_primal!!` is a deprecated Mooncake API but is the only stable
# entry point that converts a `Tangent` / `MutableTangent` back to its primal
# type.  `tangent_to_friendly!!` is the future replacement, but it does not
# yet perform the conversion for `ComponentArray` (it falls through to
# `AsRaw` and returns the raw `Tangent`).  Once `friendly_tangent_cache` is
# defined for the relevant types upstream and Mooncake removes
# `tangent_to_primal!!`, this helper should switch over.
const _MooncakeStructTangent = Union{Tangent, MutableTangent}

@inline _to_primal_alloc(primal, dx) = _copy_output(dx)
@inline function _to_primal_alloc(primal::P, dx::_MooncakeStructTangent) where {P}
    return tangent_to_primal!!(_copy_output(primal), dx)::P
end

@inline function _to_primal_into!(grad, primal, new_grad)
    copyto!(grad, new_grad)
    return grad
end
@inline function _to_primal_into!(
        grad, primal::P, new_grad::_MooncakeStructTangent
    ) where {P}
    # Build the unwrapped gradient at the *primal* type — DI allows the caller
    # to pass a `grad` buffer whose type differs from the primal (e.g. a
    # mutable `MVector` buffer for an immutable `SVector` primal), and
    # `tangent_to_primal!!` requires the destination type to match the
    # tangent's primal type.  We allocate a fresh primal-shaped buffer with
    # `_copy_output(primal)`, fill it via `tangent_to_primal!!`, then copy
    # the result into `grad`.  When `grad` itself is immutable (e.g. an
    # `SVector` buffer), no in-place update is possible — DI's `gradient!`
    # API contract cannot be honored for an immutable buffer anyway, so we
    # return the freshly built primal-shaped value, which higher-level
    # callers compare by value rather than identity.
    result = tangent_to_primal!!(_copy_output(primal), new_grad)::P
    if _can_setindex(grad)
        copyto!(grad, result)
        return grad
    else
        return result
    end
end

# Convenience used in the pullback / pushforward `foreach(_to_primal!, …)`
# call sites where there is no separate primal buffer to pass through — the
# buffer `grad` *is* the primal-shaped destination.
@inline function _to_primal!(grad, new_grad)
    copyto!(grad, new_grad)
    return grad
end
@inline function _to_primal!(grad::P, new_grad::_MooncakeStructTangent) where {P}
    return _to_primal_into!(grad, grad, new_grad)
end

# Whether `copyto!(grad, ...)` can update `grad`'s elements in place.
# `ComponentVector` is itself an immutable struct (`ismutabletype` returns
# false) but wraps a mutable `Vector`, so `copyto!` works on it; conversely,
# `SVector` wraps a `Tuple` and `copyto!` errors.  Walking down to the array
# parent and checking *its* type captures both cases correctly.
@inline _can_setindex(grad::AbstractArray) = ismutabletype(typeof(parent(grad)))
@inline _can_setindex(grad) = ismutabletype(typeof(grad))
