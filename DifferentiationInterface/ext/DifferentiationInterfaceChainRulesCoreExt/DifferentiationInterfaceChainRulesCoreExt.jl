module DifferentiationInterfaceChainRulesCoreExt

using ADTypes: ADTypes, AutoChainRules
using ChainRulesCore:
    ChainRulesCore,
    HasForwardsMode,
    HasReverseMode,
    NoTangent,
    RuleConfig,
    frule_via_ad,
    rrule_via_ad,
    unthunk
import DifferentiationInterface as DI

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

DI.check_available(::AutoChainRules) = true
DI.check_inplace(::AutoChainRules) = false
DI.check_operator_overloading(::AutoChainRules) = false

include("reverse_onearg.jl")
include("differentiate_with.jl")

end
