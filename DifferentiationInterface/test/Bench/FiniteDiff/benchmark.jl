include("../../testutils.jl")

using Pkg

using ADTypes: ADTypes
using DataFrames: DataFrame
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using Test
import Chairmarks
using StaticArrays: @SVector

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
    ) |> DataFrame
    @testset "Analyzing benchmark results" begin
        @testset "$(row[:scenario])" for row in eachrow(data)
            @test row[:allocs] == 0
        end
    end
end


@testset "allocations checks" begin
    function pushforward_allocs()
        backend = AutoFiniteDiff()
        x = @SVector [1.0, 2.0]
        tx = (2.0 .* x,)
        f(x) = @. 3.0 * x
        prep = DifferentiationInterface.prepare_pushforward(f, backend, x, tx)
        return prep
    end
    pushforward_allocs()
    allocs = @allocated prep = pushforward_allocs()
    # This needs FiniteDiff v2.31.1.
    @test allocs == 0

    function derivative_allocs()
        backend = AutoFiniteDiff()
        x = 3.0
        f(x) = x .* (@SVector [1.0, 2.0])
        prep = DifferentiationInterface.prepare_derivative(f, backend, x)
        return prep
    end
    derivative_allocs()
    allocs = @allocated prep = derivative_allocs()
    @test allocs == 0

    function jacobian_allocs()
        backend = AutoFiniteDiff()
        x = @SVector [1.0, 2.0]
        f(x) = 3.0 .* x
        prep = DifferentiationInterface.prepare_jacobian(f, backend, x)
        return prep
    end
    jacobian_allocs()
    allocs = @allocated prep = jacobian_allocs()
    # Using FiniteDiff.jl with StaticArrays to calculate a Jacobian does result in some allocations, apparently because the `FiniteDiff.JacobianCache` is a `mutable struct`.
    @test_broken allocs == 0
end
