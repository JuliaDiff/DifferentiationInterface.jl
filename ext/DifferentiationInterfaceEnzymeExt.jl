module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface
using DocStringExtensions
using Enzyme

## Forward-mode

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    _dy::Y, ::EnzymeForwardBackend, f, x::X, dx::X
) where {X,Y<:Real}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::EnzymeForwardBackend, f, x::X, dx::X
) where {X,Y<:AbstractArray}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    dy .= new_dy
    return y, dy
end

## Reverse-mode

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    _dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:Number,Y<:Union{Real,Nothing}}
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

end # module
