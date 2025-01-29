## Availability

"""
    check_available(backend)

Check whether `backend` is available (i.e. whether the extension is loaded).
"""
check_available(backend::AbstractADType) = false

function check_available(backend::SecondOrder)
    return check_available(inner(backend)) && check_available(outer(backend))
end

check_available(backend::AutoSparse) = check_available(dense_ad(backend))

function check_available(backend::MixedMode)
    return check_available(forward_backend(backend)) &&
           check_available(reverse_backend(backend))
end

## Mutation

"""
    check_inplace(backend)

Check whether `backend` supports differentiation of in-place functions.

Returns `true` or `false` in a statically predictable way.
"""
check_inplace(::AbstractADType) = true

function check_inplace(backend::SecondOrder)
    return check_inplace(inner(backend)) && check_inplace(outer(backend))
end

check_inplace(backend::AutoSparse) = check_inplace(dense_ad(backend))

function check_inplace(backend::MixedMode)
    return check_inplace(forward_backend(backend)) &&
           check_inplace(reverse_backend(backend))
end

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

function pushforward_performance(backend::Union{AutoSparse,SecondOrder})
    throw(ArgumentError("Pushforward performance not defined for $backend`."))
end

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

function pullback_performance(backend::Union{AutoSparse,SecondOrder})
    throw(ArgumentError("Pullback performance not defined for $backend`."))
end

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
    else
        return ForwardOverForward()
    end
end

function hvp_mode(backend::AutoSparse)
    throw(ArgumentError("HVP mode not defined for $backend`."))
end

## Conversions

Base.Bool(::PushforwardFast) = true
Base.Bool(::PushforwardSlow) = false

Base.Bool(::PullbackFast) = true
Base.Bool(::PullbackSlow) = false
