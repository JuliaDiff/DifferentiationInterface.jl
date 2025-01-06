using Pkg
Pkg.add(["ChainRulesCore", "Zygote"])

using ChainRulesCore
using DifferentiationInterface, DifferentiationInterfaceTest
using Test
using Zygote: ZygoteRuleConfig

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoChainRules(ZygoteRuleConfig())]
    @test check_available(backend)
    @test !check_inplace(backend)
end

test_differentiation(
    AutoChainRules(ZygoteRuleConfig()),
    default_scenarios();
    excluded=[:second_derivative],
    logging=LOGGING,
);

test_differentiation(
    AutoChainRules(ZygoteRuleConfig()),
    default_scenarios(; include_normal=false, include_constantified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
