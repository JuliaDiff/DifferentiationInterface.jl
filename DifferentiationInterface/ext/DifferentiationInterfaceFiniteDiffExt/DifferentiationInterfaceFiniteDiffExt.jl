module DifferentiationInterfaceFiniteDiffExt

using ADTypes: AutoFiniteDiff
import DifferentiationInterface as DI
using FiniteDiff:
    DerivativeCache,
    GradientCache,
    HessianCache,
    JacobianCache,
    JVPCache,
    finite_difference_derivative,
    finite_difference_gradient,
    finite_difference_gradient!,
    finite_difference_hessian,
    finite_difference_hessian!,
    finite_difference_jacobian,
    finite_difference_jacobian!,
    finite_difference_jvp,
    finite_difference_jvp!,
    default_relstep
using LinearAlgebra: dot, mul!

DI.check_available(::AutoFiniteDiff) = true

# see https://github.com/SciML/ADTypes.jl/issues/33

fdtype(::AutoFiniteDiff{fdt}) where {fdt} = fdt
fdjtype(::AutoFiniteDiff{fdt,fdjt}) where {fdt,fdjt} = fdjt
fdhtype(::AutoFiniteDiff{fdt,fdjt,fdht}) where {fdt,fdjt,fdht} = fdht

# see https://docs.sciml.ai/FiniteDiff/stable/#f-Definitions
const FUNCTION_INPLACE = Val{true}
const FUNCTION_NOT_INPLACE = Val{false}

include("onearg.jl")
include("twoarg.jl")

end # module
