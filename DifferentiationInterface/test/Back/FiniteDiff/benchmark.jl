using Pkg
Pkg.add("FiniteDiff")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

@testset "Benchmarking sparse" begin
    filtered_sparse_scenarios = filter(sparse_scenarios(; band_sizes = [])) do scen
        DIT.function_place(scen) == :in &&
            DIT.operator_place(scen) == :in &&
            scen.x isa AbstractVector &&
            scen.y isa AbstractVector
    end

    data = benchmark_differentiation(
        MyAutoSparse(AutoFiniteDiff()),
        filtered_sparse_scenarios;
        benchmark = :prepared,
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )
    @testset "Analyzing benchmark results" begin
        @testset "$(row[:scenario])" for row in eachrow(data)
            @test row[:allocs] == 0
        end
    end
end
