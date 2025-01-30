module DifferentiationInterfaceGTPSAExt

import DifferentiationInterface as DI
using ADTypes: AutoGTPSA
using GTPSA: GTPSA, TPS, Descriptor

DI.check_available(::AutoGTPSA) = true
DI.check_operator_overloading(::AutoGTPSA) = true

include("onearg.jl")
include("twoarg.jl")

end
