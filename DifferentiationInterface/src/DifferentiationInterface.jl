"""
    DifferentiationInterface

An interface to various automatic differentiation backends in Julia.
"""
module DifferentiationInterface

using ADTypes:
    ADTypes,
    AbstractADType,
    AbstractSparsityDetector,
    AutoSparse,
    ForwardMode,
    ForwardOrReverseMode,
    ReverseMode,
    SymbolicMode,
    dense_ad,
    mode,
    jacobian_sparsity,
    hessian_sparsity
using ADTypes:
    AutoChainRules,
    AutoDiffractor,
    AutoEnzyme,
    AutoFastDifferentiation,
    AutoFiniteDiff,
    AutoFiniteDifferences,
    AutoForwardDiff,
    AutoGTPSA,
    AutoMooncake,
    AutoPolyesterForwardDiff,
    AutoReverseDiff,
    AutoSymbolics,
    AutoTracker,
    AutoZygote
using LinearAlgebra: dot

include("compat.jl")
include("docstrings.jl")

include("first_order/mixed_mode.jl")
include("second_order/second_order.jl")

include("utils/context.jl")
include("utils/prep.jl")
include("utils/traits.jl")
include("utils/basis.jl")
include("utils/batchsize.jl")
include("utils/check.jl")
include("utils/errors.jl")
include("utils/linalg.jl")
include("utils/sparse.jl")

include("first_order/pushforward.jl")
include("first_order/pullback.jl")
include("first_order/derivative.jl")
include("first_order/gradient.jl")
include("first_order/jacobian.jl")

include("second_order/second_derivative.jl")
include("second_order/hvp.jl")
include("second_order/hessian.jl")

include("fallbacks/no_prep.jl")
include("fallbacks/change_prep.jl")

include("misc/differentiate_with.jl")
include("misc/from_primitive.jl")
include("misc/sparsity_detector.jl")
include("misc/simple_finite_diff.jl")
include("misc/zero_backends.jl")
include("misc/overloading.jl")

## Exported

export Context, Constant, Cache, ConstantOrCache
export MixedMode, SecondOrder

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback

export value_and_derivative!, value_and_derivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

export pushforward!, pushforward
export pullback!, pullback

export derivative!, derivative
export gradient!, gradient
export jacobian!, jacobian

export second_derivative!, second_derivative
export value_derivative_and_second_derivative, value_derivative_and_second_derivative!
export hvp!, hvp, gradient_and_hvp, gradient_and_hvp!
export hessian!, hessian
export value_gradient_and_hessian, value_gradient_and_hessian!

export prepare_pushforward, prepare_pushforward_same_point
export prepare_pullback, prepare_pullback_same_point
export prepare_hvp, prepare_hvp_same_point
export prepare_derivative, prepare_gradient, prepare_jacobian
export prepare_second_derivative, prepare_hessian

export DifferentiateWith
export DenseSparsityDetector

export check_available, check_inplace

## Re-exported from ADTypes

export AutoChainRules
export AutoDiffractor
export AutoEnzyme
export AutoFastDifferentiation
export AutoFiniteDiff
export AutoFiniteDifferences
export AutoForwardDiff
export AutoGTPSA
export AutoMooncake
export AutoPolyesterForwardDiff
export AutoReverseDiff
export AutoSymbolics
export AutoTracker
export AutoZygote

export AutoSparse

## Public but not exported

@public inner, outer

include("init.jl")

end # module
