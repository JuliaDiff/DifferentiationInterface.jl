abstract type Prep end

"""
$(docstring_preptype("PushforwardPrep", "pushforward"))
"""
abstract type PushforwardPrep <: Prep end
struct NoPushforwardPrep <: PushforwardPrep end

"""
$(docstring_preptype("PullbackPrep", "pullback"))
"""
abstract type PullbackPrep <: Prep end
struct NoPullbackPrep <: PullbackPrep end

"""
$(docstring_preptype("DerivativePrep", "derivative"))
"""
abstract type DerivativePrep <: Prep end
struct NoDerivativePrep <: DerivativePrep end

"""
$(docstring_preptype("GradientPrep", "gradient"))
"""
abstract type GradientPrep <: Prep end
struct NoGradientPrep <: GradientPrep end

"""
$(docstring_preptype("JacobianPrep", "jacobian"))
"""
abstract type JacobianPrep <: Prep end
struct NoJacobianPrep <: JacobianPrep end

"""
$(docstring_preptype("HVPPrep", "hvp"))
"""
abstract type HVPPrep <: Prep end
struct NoHVPPrep <: HVPPrep end

"""
$(docstring_preptype("HessianPrep", "hessian"))
"""
abstract type HessianPrep <: Prep end
struct NoHessianPrep <: HessianPrep end

"""
$(docstring_preptype("SecondDerivativePrep", "second_derivative"))
"""
abstract type SecondDerivativePrep <: Prep end
struct NoSecondDerivativePrep <: SecondDerivativePrep end
