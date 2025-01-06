module DifferentiationInterfaceGTPSAExt

import DifferentiationInterface as DI
using ADTypes: AutoGTPSA
using GTPSA

DI.check_available(::AutoGTPSA) = true

include("utils.jl")
include("onearg.jl")
#include("twoarg.jl")

end
