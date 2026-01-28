using Pkg; Pkg.activate(@__DIR__)

using ADTypes
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Tables, DataAPI
using Test

row1 = DIT.DifferentiationBenchmarkDataRow(;
    backend = AutoForwardDiff(),
    scenario = Scenario{:gradient, :out}(sum, ones(2)),
    operator = :gradient,
    prepared = true,
    calls = 2,
    samples = 1,
    evals = 1,
    time = 1.0,
    allocs = 10.0,
    bytes = 100.0,
    gc_fraction = 0.5,
    compile_fraction = 0.1
)

tab = DIT.DifferentiationBenchmark([row1, row1])

@testset "Tables API"  begin
    @test Tables.istable(typeof(tab))
    @test Tables.rowaccess(typeof(tab))
    @test Tables.columnaccess(typeof(tab))
    @test DataAPI.nrow(tab) == 2
    @test DataAPI.ncol(tab) == 12
    @test Tables.rows(tab) == tab.rows
    @test Tables.columns(tab) == tab
    @test Tables.getcolumn(tab, :samples) == [1, 1]
    @test Tables.getcolumn(row1, :samples) == 1
    @test Tables.getcolumn(tab, 5) == [2, 2]
    @test Tables.getcolumn(row1, 5) == 2
    @test Tables.columnnames(tab) |> length == 12
    @test Tables.columnnames(row1) |> length == 12
end
