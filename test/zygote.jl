using DifferentiationInterface
using Zygote

test_pullback(ChainRulesReverseBackend(Zygote.ZygoteRuleConfig()); type_stability=false);
test_jacobian_and_friends(
    ChainRulesReverseBackend(Zygote.ZygoteRuleConfig()); type_stability=false
);
