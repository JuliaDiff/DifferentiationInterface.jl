abstract type Prep end

"""
    PushforwardPrep

Abstract type for additional information needed by [`pushforward`](@ref) and its variants.
"""
abstract type PushforwardPrep <: Prep end
struct NoPushforwardPrep <: PushforwardPrep end

"""
    PullbackPrep

Abstract type for additional information needed by [`pullback`](@ref) and its variants.
"""
abstract type PullbackPrep <: Prep end
struct NoPullbackPrep <: PullbackPrep end

"""
    DerivativePrep

Abstract type for additional information needed by [`derivative`](@ref) and its variants.
"""
abstract type DerivativePrep <: Prep end
struct NoDerivativePrep <: DerivativePrep end

"""
    GradientPrep

Abstract type for additional information needed by [`gradient`](@ref) and its variants.
"""
abstract type GradientPrep <: Prep end
struct NoGradientPrep <: GradientPrep end

"""
    JacobianPrep

Abstract type for additional information needed by [`jacobian`](@ref) and its variants.
"""
abstract type JacobianPrep <: Prep end
struct NoJacobianPrep <: JacobianPrep end

"""
    HVPPrep

Abstract type for additional information needed by [`hvp`](@ref) and its variants.
"""
abstract type HVPPrep <: Prep end
struct NoHVPPrep <: HVPPrep end

"""
    HessianPrep

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianPrep <: Prep end
struct NoHessianPrep <: HessianPrep end

"""
    SecondDerivativePrep

Abstract type for additional information needed by [`second_derivative`](@ref) and its variants.
"""
abstract type SecondDerivativePrep <: Prep end
struct NoSecondDerivativePrep <: SecondDerivativePrep end
