"""
    MixedMode

Combination of a forward and a reverse mode backend for mixed-mode Jacobian computation.

!!! danger
    `MixedMode` backends only support [`jacobian`](@ref) and its variants.

# Constructor

    MixedMode(forward_backend, reverse_backend)
"""
struct MixedMode{F<:AbstractADType,R<:AbstractADType} <: AbstractADType
    forward::F
    reverse::R
    function MixedMode(forward::AbstractADType, reverse::AbstractADType)
        @assert pushforward_performance(forward) isa PushforwardFast
        @assert pullback_performance(reverse) isa PullbackFast
        return new{typeof(forward),typeof(reverse)}(forward, reverse)
    end
end

forward_backend(m::MixedMode) = m.forward
reverse_backend(m::MixedMode) = m.reverse

struct ForwardAndReverseMode <: ADTypes.AbstractMode end
ADTypes.mode(::MixedMode) = ForwardAndReverseMode()
