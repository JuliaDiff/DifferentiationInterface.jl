module DifferentiationInterfaceGTPSAExt

import DifferentiationInterface as DI
using ADTypes: AutoGTPSA
using GTPSA
using LinearAlgebra

DI.check_available(::AutoGTPSA) = true

include("onearg.jl")
include("twoarg.jl")

end
