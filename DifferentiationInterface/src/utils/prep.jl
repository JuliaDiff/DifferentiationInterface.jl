abstract type Prep{SIG} end

"""
    PushforwardPrep{SIG}

Abstract type for additional information needed by [`pushforward`](@ref) and its variants.
"""
abstract type PushforwardPrep{SIG} <: Prep{SIG} end
struct NoPushforwardPrep{SIG} <: PushforwardPrep{SIG} end

"""
    PullbackPrep{SIG}

Abstract type for additional information needed by [`pullback`](@ref) and its variants.
"""
abstract type PullbackPrep{SIG} <: Prep{SIG} end
struct NoPullbackPrep{SIG} <: PullbackPrep{SIG} end

"""
    DerivativePrep{SIG}

Abstract type for additional information needed by [`derivative`](@ref) and its variants.
"""
abstract type DerivativePrep{SIG} <: Prep{SIG} end
struct NoDerivativePrep{SIG} <: DerivativePrep{SIG} end

"""
    GradientPrep{SIG}

Abstract type for additional information needed by [`gradient`](@ref) and its variants.
"""
abstract type GradientPrep{SIG} <: Prep{SIG} end
struct NoGradientPrep{SIG} <: GradientPrep{SIG} end

"""
    JacobianPrep{SIG}

Abstract type for additional information needed by [`jacobian`](@ref) and its variants.
"""
abstract type JacobianPrep{SIG} <: Prep{SIG} end
struct NoJacobianPrep{SIG} <: JacobianPrep{SIG} end

"""
    HVPPrep{SIG}

Abstract type for additional information needed by [`hvp`](@ref) and its variants.
"""
abstract type HVPPrep{SIG} <: Prep{SIG} end
struct NoHVPPrep{SIG} <: HVPPrep{SIG} end

"""
    HessianPrep{SIG}

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianPrep{SIG} <: Prep{SIG} end
struct NoHessianPrep{SIG} <: HessianPrep{SIG} end

"""
    SecondDerivativePrep{SIG}

Abstract type for additional information needed by [`second_derivative`](@ref) and its variants.
"""
abstract type SecondDerivativePrep{SIG} <: Prep{SIG} end
struct NoSecondDerivativePrep{SIG} <: SecondDerivativePrep{SIG} end

function check_prep(
    f, prep::Prep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {C}
end
