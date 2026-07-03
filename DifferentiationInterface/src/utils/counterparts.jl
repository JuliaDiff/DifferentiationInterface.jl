"""
    forward_counterpart(backend)

Return a forward-mode counterpart of `backend`.

If `backend` has a dedicated forward-mode counterpart (e.g. `AutoMooncake` has
`AutoMooncakeForward`), it is returned. Else, `backend` itself is returned (DI allows
reverse-mode backends to execute pushforwards), with a warning in the case it is
reverse-mode only.
"""
function forward_counterpart(backend::AbstractADType)
    if !(mode(backend) isa Union{ForwardMode, ForwardOrReverseMode, SymbolicMode})
        @warn "The forward-mode counterpart of `$backend` is itself, returning it unchanged." maxlog =
            1
    end
    return backend
end

"""
    reverse_counterpart(backend)

Return a reverse-mode counterpart of `backend`.

If `backend` has a dedicated reverse-mode counterpart (e.g. `AutoMooncakeForward` has
`AutoMooncake`), it is returned. Else, `backend` itself is returned (DI allows
forward-mode backends to execute pullbacks), with a warning in the case it is
forward-mode only.
"""
function reverse_counterpart(backend::AbstractADType)
    if !(mode(backend) isa Union{ReverseMode, ForwardOrReverseMode, SymbolicMode})
        @warn "The reverse-mode counterpart of `$backend` is itself, returning it unchanged." maxlog =
            1
    end
    return backend
end

## Wrapper backends

# The counterpart of `AutoSparse` acts on the dense backend and preserves the sparsity
# detection and coloring machinery.

function forward_counterpart(backend::AutoSparse)
    return AutoSparse(
        forward_counterpart(dense_ad(backend));
        sparsity_detector = backend.sparsity_detector,
        coloring_algorithm = backend.coloring_algorithm,
    )
end

function reverse_counterpart(backend::AutoSparse)
    return AutoSparse(
        reverse_counterpart(dense_ad(backend));
        sparsity_detector = backend.sparsity_detector,
        coloring_algorithm = backend.coloring_algorithm,
    )
end

# A `MixedMode` backend already combines a forward- and a reverse-mode backend, each used
# where it works best, so it is its own counterpart in both directions.

forward_counterpart(backend::MixedMode) = backend
reverse_counterpart(backend::MixedMode) = backend

# For `SecondOrder` there is no meaningful counterpart: flipping the modes of the inner
# and outer backends changes the nature of the second-order combination.

function forward_counterpart(::SecondOrder)
    throw(
        ArgumentError(
            "`forward_counterpart` is ambiguous for `SecondOrder` backends, apply it to `inner(backend)` and `outer(backend)` separately.",
        )
    )
end

function reverse_counterpart(::SecondOrder)
    throw(
        ArgumentError(
            "`reverse_counterpart` is ambiguous for `SecondOrder` backends, apply it to `inner(backend)` and `outer(backend)` separately.",
        )
    )
end

# Counterparts for `FromPrimitive` wrappers are defined in `misc/from_primitive.jl`, and
# backend-specific counterparts (e.g., for AutoMooncake and AutoEnzyme) are defined in the
# corresponding package extensions.
