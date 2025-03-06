abstract type Prep{F,Y,B<:AbstractADType,X,T<:Union{NTuple,Nothing},CC} end

"""
    PushforwardPrep

Abstract type for additional information needed by [`pushforward`](@ref) and its variants.
"""
abstract type PushforwardPrep{F,Y,B,X,T,CC} <: Prep{F,Y,B,X,T,CC} end
struct NoPushforwardPrep{F,Y,B,X,T,CC} <: PushforwardPrep{F,Y,B,X,T,CC} end

"""
    PullbackPrep

Abstract type for additional information needed by [`pullback`](@ref) and its variants.
"""
abstract type PullbackPrep{F,Y,B,X,T,CC} <: Prep{F,Y,B,X,T,CC} end
struct NoPullbackPrep{F,Y,B,X,T,CC} <: PullbackPrep{F,Y,B,X,T,CC} end

"""
    DerivativePrep

Abstract type for additional information needed by [`derivative`](@ref) and its variants.
"""
abstract type DerivativePrep{F,Y,B,X,CC} <: Prep{F,Y,B,X,Nothing,CC} end
struct NoDerivativePrep{F,Y,B,X,CC} <: DerivativePrep{F,Y,B,X,Nothing,CC} end

"""
    GradientPrep

Abstract type for additional information needed by [`gradient`](@ref) and its variants.
"""
abstract type GradientPrep{F,B,X,CC} <: Prep{F,Nothing,B,X,Nothing,CC} end
struct NoGradientPrep{F,B,X,CC} <: GradientPrep{F,Nothing,B,X,Nothing,CC} end

"""
    JacobianPrep

Abstract type for additional information needed by [`jacobian`](@ref) and its variants.
"""
abstract type JacobianPrep{F,Y,B,X,CC} <: Prep{F,Y,B,X,Nothing,CC} end
struct NoJacobianPrep{F,Y,B,X,CC} <: JacobianPrep{F,Y,B,X,Nothing,CC} end

"""
    HVPPrep

Abstract type for additional information needed by [`hvp`](@ref) and its variants.
"""
abstract type HVPPrep{F,B,X,CC} <: Prep{F,B,X,Nothing,CC} end
struct NoHVPPrep{F,B,X,CC} <: HVPPrep{F,B,X,Nothing,CC} end

"""
    HessianPrep

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianPrep{F,B,X,CC} <: Prep{F,Nothing,B,X,Nothing,CC} end
struct NoHessianPrep{F,B,X,CC} <: HessianPrep{F,Nothing,B,X,Nothing,CC} end

"""
    SecondDerivativePrep

Abstract type for additional information needed by [`second_derivative`](@ref) and its variants.
"""
abstract type SecondDerivativePrep{F,B,X,CC} <: Prep{F,Nothing,B,X,Nothing,CC} end
struct NoSecondDerivativePrep{F,B,X,CC} <: SecondDerivativePrep{F,Nothing,B,X,Nothing,CC} end

function check_prep(
    f, ::Prep{F,Y,B,X,T,CC}, backend, x, contexts::Vararg{Context,C}
) where {F,Y,B,X,T,CC,C}
    @assert typeof(f) == F
    @assert Nothing == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert Nothing == T
    @assert typeof(contexts) == CC
end

function check_prep(
    f, ::Prep{F,Y,B,X,T,CC}, backend, x, t::NTuple, contexts::Vararg{Context,C}
) where {F,Y,B,X,T,CC,C}
    @assert typeof(f) == F
    @assert Nothing == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert typeof(t) == T
    @assert typeof(contexts) == CC
end

function check_prep(
    f, y, ::Prep{F,Y,B,X,T,CC}, backend, x, contexts::Vararg{Context,C}
) where {F,Y,B,X,T,CC,C}
    @assert typeof(f) == F
    @assert typeof(y) == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert Nothing == T
    @assert typeof(contexts) == CC
end

function check_prep(
    f, y, ::Prep{F,Y,B,X,CC}, backend, x, t::NTuple, contexts::Vararg{Context,C}
) where {F,Y,B,X,CC,C}
    @assert typeof(f) == F
    @assert typeof(y) == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert typeof(t) == T
    @assert typeof(contexts) == CC
end
