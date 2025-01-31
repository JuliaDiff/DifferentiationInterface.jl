using Pkg
Pkg.add("GTPSA")

using DifferentiationInterface, DifferentiationInterfaceTest
using GTPSA: GTPSA
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

# Test no Descriptor (use context)
test_differentiation(
    AutoGTPSA(),
    default_scenarios(; include_constantified=true);
    type_stability=:full,
    logging=LOGGING,
);

# Test with Descriptor:
d1 = GTPSA.Descriptor(20, 2) # 20 variables to 2nd order
test_differentiation(AutoGTPSA(d1); type_stability=:full, logging=LOGGING);

# Test with Descriptor using varying orders 
vos = 2 * ones(Int, 20)
vos[1] = 3
d2 = GTPSA.Descriptor(vos, 3)
test_differentiation(AutoGTPSA(d2); type_stability=:full, logging=LOGGING);
