module DifferentiationInterfaceEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    JacobianExtras,
    HVPExtras,
    PullbackExtras,
    PushforwardExtras,
    NoDerivativeExtras,
    NoGradientExtras,
    NoHVPExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    NoPushforwardExtras,
    Tangents,
    pick_batchsize
using Enzyme:
    Active,
    Annotation,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    EnzymeCore,
    Forward,
    ForwardMode,
    MixedDuplicated,
    Mode,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitWithPrimal,
    ReverseMode,
    autodiff,
    autodiff_deferred,
    autodiff_deferred_thunk,
    autodiff_thunk,
    chunkedonehot,
    gradient,
    gradient!,
    guess_activity,
    hvp,
    hvp!,
    jacobian,
    make_zero,
    make_zero!,
    onehot

include("utils.jl")

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

include("second_order.jl")

end # module
