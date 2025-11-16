using Pkg
Pkg.add("ForwardDiff")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using StaticArrays: StaticArrays, @SVector
using Test

LOGGING = get(ENV, "CI", "false") == "false"

@info "Printing allocs"
const f! = DIT.diffsquare!
x, y = rand(10), zeros(9);
backend = MyAutoSparse(AutoForwardDiff());
prep = prepare_jacobian(f!, y, backend, x);
J = similar(sparsity_pattern(prep), Float64)
jacobian!(f!, y, J, prep, backend, x)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1 jacobian!(f!, y, J, prep, backend, x)
Profile.Allocs.print()

@info "Printing allocs done"

@testset verbose = true "Benchmarking static" begin
    filtered_static_scenarios = filter(static_scenarios(; include_batchified = false)) do scen
        DIT.function_place(scen) == :out && DIT.operator_place(scen) == :out
    end
    data = benchmark_differentiation(
        AutoForwardDiff(),
        filtered_static_scenarios;
        benchmark = :prepared,
        excluded = [:hessian, :pullback],  # TODO: figure this out
        logging = LOGGING,
    )
    @testset "Analyzing benchmark results" begin
        @testset "$(row[:scenario])" for row in eachrow(data)
            @test row[:allocs] == 0
        end
    end
end

@testset "Benchmarking sparse" begin
    filtered_sparse_scenarios = filter(sparse_scenarios(; band_sizes = [])) do scen
        DIT.function_place(scen) == :in &&
            DIT.operator_place(scen) == :in &&
            scen.x isa AbstractVector &&
            scen.y isa AbstractVector
    end

    data = benchmark_differentiation(
        MyAutoSparse(AutoForwardDiff()),
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
