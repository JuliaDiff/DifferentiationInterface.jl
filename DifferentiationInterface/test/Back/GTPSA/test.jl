using Pkg
Pkg.add("GTPSA")

using DifferentiationInterface, DifferentiationInterfaceTest
using GTPSA: GTPSA
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoGTPSA()]
    @test check_available(backend)
    @test check_inplace(backend)
end

# Test no Descriptor (use context)
test_differentiation(AutoGTPSA(); type_stability=:full, logging=LOGGING);

# Test with Descriptor:
d1 = Descriptor(20, 2) # 20 variables to 2nd order
test_differentiation(AutoGTPSA(d1); type_stability=:full, logging=LOGGING);

# Test with Descriptor using varying orders 
vos = 2*ones(Int, 20)
vos[1] = 3
d2 = Descriptor(vos, 3)
test_differentiation(AutoGTPSA(d2); type_stability=:full, logging=LOGGING);
