"""
    forward_counterpart(backend)

Return a forward-mode counterpart of `backend`.

If `backend` has a dedicated forward-mode counterpart (e.g. `AutoMooncake` has
`AutoMooncakeForward`), it is returned. Else, a warning is emitted in the case the `backend`
is reverse-mode only.
"""
function forward_counterpart(backend::AbstractADType)
    if !(mode(backend) isa Union{ForwardMode, ForwardOrReverseMode, SymbolicMode})
        @warn "No forward-mode counterpart known for `$backend`, returning it unchanged." maxlog =
            1
    end
    return backend
end

"""
    reverse_counterpart(backend)

Return a reverse-mode counterpart of `backend`.

If `backend` has a dedicated reverse-mode counterpart (e.g. `AutoMooncakeForward` has
`AutoMooncake`), it is returned. Else, a warning is emitted in the case the `backend` is
forward-mode only.
"""
function reverse_counterpart(backend::AbstractADType)
    if !(mode(backend) isa Union{ReverseMode, ForwardOrReverseMode, SymbolicMode})
        @warn "No reverse-mode counterpart known for `$backend`, returning it unchanged." maxlog =
            1
    end
    return backend
end

# Backend-specific counterparts (e.g., for AutoMooncake and AutoEnzyme) are defined in the
# corresponding package extensions.
