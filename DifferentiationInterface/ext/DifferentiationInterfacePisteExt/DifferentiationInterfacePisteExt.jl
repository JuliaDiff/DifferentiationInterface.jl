module DifferentiationInterfacePisteExt

using ADTypes: AutoPiste, ForwardMode, ReverseMode
import DifferentiationInterface as DI
using Piste:
    Piste,
    Dual,
    GradientWorkspace,
    ReverseWorkspace,
    TapedReal,
    gradient!,
    pickchunk,
    rev_value_and_pullback,
    rev_value_and_pullback!,
    value,
    value_and_gradient!

DI.check_available(::AutoPiste) = true
DI.inplace_support(::AutoPiste) = DI.InPlaceSupported()

include("forward.jl")
include("reverse.jl")

end # module
