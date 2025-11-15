module DifferentiationInterfaceReactantExt

using ADTypes: ADTypes, AutoReactant
import DifferentiationInterface as DI
using Reactant: @compile, to_rarray

DI.check_available(backend::AutoReactant) = DI.check_available(backend.mode)
DI.inplace_support(backend::AutoReactant) = DI.inplace_support(backend.mode)

include("onearg.jl")

end # module
