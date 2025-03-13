module DifferentiationInterfaceForwardDiffExt

using ADTypes: AutoForwardDiff
import DifferentiationInterface as DI
import DiffResults as DR
using DiffResults: DiffResults, DiffResult, GradientResult, HessianResult, MutableDiffResult
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    ForwardDiff,
    GradientConfig,
    HessianConfig,
    JacobianConfig,
    Tag,
    checktag,
    derivative,
    derivative!,
    extract_derivative,
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
    partials,
    value

DI.check_available(::AutoForwardDiff) = true

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")
include("secondorder.jl")
include("differentiate_with.jl")
include("misc.jl")

end # module
