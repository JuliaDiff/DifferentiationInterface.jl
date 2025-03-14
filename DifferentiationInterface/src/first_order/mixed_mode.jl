"""
    MixedMode

Combination of a forward and a reverse mode backend for mixed-mode sparse Jacobian computation.

!!! danger
    `MixedMode` backends only support [`jacobian`](@ref) and its variants, and it should be used inside an [`AutoSparse`](@extref ADTypes.AutoSparse) wrapper.

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

"""
    forward_backend(m::MixedMode)

Return the forward-mode part of a `MixedMode` backend.
"""
forward_backend(m::MixedMode) = m.forward

"""
    reverse_backend(m::MixedMode)

Return the reverse-mode part of a `MixedMode` backend.
"""
reverse_backend(m::MixedMode) = m.reverse

"""
    ForwardAndReverseMode <: ADTypes.AbstractMode

Appropriate mode type for `MixedMode` backends.
"""
struct ForwardAndReverseMode <: ADTypes.AbstractMode end
ADTypes.mode(::MixedMode) = ForwardAndReverseMode()
