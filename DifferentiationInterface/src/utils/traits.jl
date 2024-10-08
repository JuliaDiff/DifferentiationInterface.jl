## Mutation

abstract type InPlaceBehavior end

"""
    InPlaceSupported

Trait identifying backends that support in-place functions `f!(y, x)`.
"""
struct InPlaceSupported <: InPlaceBehavior end

"""
    InPlaceNotSupported

Trait identifying backends that do not support in-place functions `f!(y, x)`.
"""
struct InPlaceNotSupported <: InPlaceBehavior end

"""
    inplace_support(backend)

Return [`InPlaceSupported`](@ref) or [`InPlaceNotSupported`](@ref) in a statically predictable way.
"""
inplace_support(::AbstractADType) = InPlaceSupported()

function inplace_support(backend::SecondOrder)
    if Bool(inplace_support(inner(backend))) && Bool(inplace_support(outer(backend)))
        return InPlaceSupported()
    else
        return InPlaceNotSupported()
    end
end

inplace_support(backend::AutoSparse) = inplace_support(dense_ad(backend))

## Pushforward

abstract type PushforwardPerformance end

"""
    PushforwardFast

Trait identifying backends that support efficient pushforwards.
"""
struct PushforwardFast <: PushforwardPerformance end

"""
    PushforwardSlow

Trait identifying backends that do not support efficient pushforwards.
"""
struct PushforwardSlow <: PushforwardPerformance end

"""
    pushforward_performance(backend)

Return [`PushforwardFast`](@ref) or [`PushforwardSlow`](@ref) in a statically predictable way.
"""
pushforward_performance(backend::AbstractADType) = pushforward_performance(mode(backend))
pushforward_performance(::ForwardMode) = PushforwardFast()
pushforward_performance(::ForwardOrReverseMode) = PushforwardFast()
pushforward_performance(::ReverseMode) = PushforwardSlow()
pushforward_performance(::SymbolicMode) = PushforwardFast()
pushforward_performance(backend::AutoSparse) = pushforward_performance(dense_ad(backend))

## Pullback

abstract type PullbackPerformance end

"""
    PullbackFast

Trait identifying backends that support efficient pullbacks.
"""
struct PullbackFast <: PullbackPerformance end

"""
    PullbackSlow

Trait identifying backends that do not support efficient pullbacks.
"""
struct PullbackSlow <: PullbackPerformance end

"""
    pullback_performance(backend)

Return [`PullbackFast`](@ref) or [`PullbackSlow`](@ref) in a statically predictable way.
"""
pullback_performance(backend::AbstractADType) = pullback_performance(mode(backend))
pullback_performance(::ForwardMode) = PullbackSlow()
pullback_performance(::ForwardOrReverseMode) = PullbackFast()
pullback_performance(::ReverseMode) = PullbackFast()
pullback_performance(::SymbolicMode) = PullbackFast()
pullback_performance(backend::AutoSparse) = pullback_performance(dense_ad(backend))

## HVP

abstract type HVPMode end

"""
    ForwardOverReverse

Traits identifying second-order backends that compute HVPs in forward over reverse mode.
"""
struct ForwardOverReverse <: HVPMode end

"""
    ReverseOverForward

Traits identifying second-order backends that compute HVPs in reverse over forward mode.
"""
struct ReverseOverForward <: HVPMode end

"""
    ReverseOverReverse

Traits identifying second-order backends that compute HVPs in reverse over reverse mode.
"""
struct ReverseOverReverse <: HVPMode end

"""
    ForwardOverForward

Traits identifying second-order backends that compute HVPs in forward over forward mode (inefficient).
"""
struct ForwardOverForward <: HVPMode end

hvp_mode(backend::AbstractADType) = hvp_mode(SecondOrder(backend, backend))

function hvp_mode(ba::SecondOrder)
    if Bool(pushforward_performance(outer(ba))) && Bool(pullback_performance(inner(ba)))
        return ForwardOverReverse()
    elseif Bool(pullback_performance(outer(ba))) && Bool(pushforward_performance(inner(ba)))
        return ReverseOverForward()
    elseif Bool(pullback_performance(outer(ba))) && Bool(pullback_performance(inner(ba)))
        return ReverseOverReverse()
    elseif Bool(pushforward_performance(outer(ba))) &&
        Bool(pushforward_performance(inner(ba)))
        return ForwardOverForward()
    end
end

hvp_mode(backend::AutoSparse{<:SecondOrder}) = hvp_mode(dense_ad(backend))

## Conversions

Base.Bool(::InPlaceSupported) = true
Base.Bool(::InPlaceNotSupported) = false

Base.Bool(::PushforwardFast) = true
Base.Bool(::PushforwardSlow) = false

Base.Bool(::PullbackFast) = true
Base.Bool(::PullbackSlow) = false
