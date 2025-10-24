using DifferentiationInterface: Prep
using InteractiveUtils: subtypes
using Test

@test subtypes(Prep) == [
    DifferentiationInterface.DerivativePrep,
    DifferentiationInterface.GradientPrep,
    DifferentiationInterface.HVPPrep,
    DifferentiationInterface.HessianPrep,
    DifferentiationInterface.JacobianPrep,
    DifferentiationInterface.PullbackPrep,
    DifferentiationInterface.PushforwardPrep,
    DifferentiationInterface.SecondDerivativePrep,
]
