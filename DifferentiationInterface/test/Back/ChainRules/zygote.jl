using Pkg
Pkg.add(["ChainRulesCore", "Zygote"])

using ChainRulesCore
using DifferentiationInterface, DifferentiationInterfaceTest
using Test
using Zygote: ZygoteRuleConfig

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

test_differentiation(
    AutoChainRules(ZygoteRuleConfig()),
    default_scenarios();
    excluded=SECOND_ORDER,
    logging=LOGGING,
);

test_differentiation(
    AutoChainRules(ZygoteRuleConfig()),
    default_scenarios(; include_normal=false, include_constantified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
