using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: default_scenarios
using ForwardDiff: ForwardDiff
using Test

@testset "Naming" begin
    scen = Scenario{:gradient, :out}(
        sum, zeros(10); res1 = ones(10), name = "My pretty little scenario"
    )
    @test string(scen) == "My pretty little scenario"

    testset = test_differentiation(
        AutoForwardDiff(), [scen]; testset_name = "My amazing test set"
    )

    data = benchmark_differentiation(
        AutoForwardDiff(), [scen]; testset_name = "My amazing test set"
    )
end;

@testset "Compute results" begin
    scens = default_scenarios()
    new_scens = map(s -> compute_results(s, AutoForwardDiff()), scens)

    isapprox_robust(x, y) = isapprox(x, y)
    isapprox_robust(x::Nothing, y::Nothing) = true
    isapprox_robust(x::NTuple, y::NTuple) = all(map(isapprox, x, y))

    for (sa, sb) in zip(scens, new_scens)
        @test isapprox_robust(sa.res1, sb.res1)
        @test isapprox_robust(sa.res2, sb.res2)
    end
end;
