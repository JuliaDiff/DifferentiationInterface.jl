module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff: Dual, Tag, value, extract_derivative, extract_derivative!
using LinearAlgebra: mul!

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:Real,Y<:Real}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:Real,Y<:AbstractArray}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:AbstractArray,Y<:Real}
    T = typeof(Tag(f, X))  # TODO: unsure
    xdual = Dual{T}.(x, dx)  # TODO: allocation
    ydual = f(xdual)
    y = value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx
) where {X<:AbstractArray,Y<:AbstractArray}
    T = typeof(Tag(f, X))  # TODO: unsure
    xdual = Dual{T}.(x, dx)  # TODO: allocation
    ydual = f(xdual)
    y = value.(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

end # module
