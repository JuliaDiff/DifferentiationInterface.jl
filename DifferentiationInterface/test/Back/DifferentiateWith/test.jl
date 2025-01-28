using Pkg
Pkg.add(["FiniteDiff", "ForwardDiff", "Zygote"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Test

LOGGING = get(ENV, "CI", "false") == "false"

function differentiatewith_scenarios()
    bad_scens =  # these closurified scenarios have mutation and type constraints
        filter(default_scenarios(; include_normal=false, include_closurified=true)) do scen
            DIT.function_place(scen) == :out
        end
    good_scens = map(bad_scens) do scen
        DIT.change_function(
            scen, DifferentiateWith(scen.f, AutoFiniteDiff()); keep_smaller=false
        )
    end
    return good_scens
end

test_differentiation(
    [AutoForwardDiff(), AutoZygote()],
    differentiatewith_scenarios();
    excluded=SECOND_ORDER,
    logging=LOGGING,
)
