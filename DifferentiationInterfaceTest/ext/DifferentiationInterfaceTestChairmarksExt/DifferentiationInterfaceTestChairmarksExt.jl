module DifferentiationInterfaceTestChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be, Benchmark, Sample
import DifferentiationInterface as DI
using DifferentiationInterface:
    prepare_pushforward,
    prepare_pushforward_same_point,
    prepare!_pushforward,
    pushforward,
    pushforward!,
    value_and_pushforward,
    value_and_pushforward!,
    prepare_pullback,
    prepare_pullback_same_point,
    prepare!_pullback,
    pullback,
    pullback!,
    value_and_pullback,
    value_and_pullback!,
    prepare_derivative,
    prepare!_derivative,
    derivative,
    derivative!,
    value_and_derivative,
    value_and_derivative!,
    prepare_gradient,
    prepare!_gradient,
    gradient,
    gradient!,
    value_and_gradient,
    value_and_gradient!,
    prepare_jacobian,
    prepare!_jacobian,
    jacobian,
    jacobian!,
    value_and_jacobian,
    value_and_jacobian!,
    prepare_second_derivative,
    prepare!_second_derivative,
    second_derivative,
    second_derivative!,
    value_derivative_and_second_derivative,
    value_derivative_and_second_derivative!,
    prepare_hvp,
    prepare_hvp_same_point,
    prepare!_hvp,
    hvp,
    hvp!,
    gradient_and_hvp,
    gradient_and_hvp!,
    prepare_hessian,
    prepare!_hessian,
    hessian,
    hessian!,
    value_gradient_and_hessian,
    value_gradient_and_hessian!
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest:
    ALL_OPS,
    CallCounter, CallsResult, DifferentiationBenchmarkDataRow, DifferentiationBenchmark, Scenario,
    mysimilar, reset_count!
using Test

function failed_bench()
    evals = 0.0
    time = NaN
    allocs = NaN
    bytes = NaN
    gc_fraction = NaN
    compile_fraction = NaN
    recompile_fraction = NaN
    warmup = NaN
    checksum = NaN
    sample = Sample(
        evals,
        time,
        allocs,
        bytes,
        gc_fraction,
        compile_fraction,
        recompile_fraction,
        warmup,
        checksum,
    )
    return Benchmark([sample])
end

@kwdef struct BenchmarkResult
    prepared_valop::Benchmark = failed_bench()
    prepared_op::Benchmark = failed_bench()
    preparation::Benchmark = failed_bench()
    unprepared_valop::Benchmark = failed_bench()
    unprepared_op::Benchmark = failed_bench()
end


function record!(
        data::DifferentiationBenchmark;
        backend::AbstractADType,
        scenario::Scenario,
        operator::String,
        prepared::Union{Nothing, Bool},
        bench::Benchmark,
        calls::Integer,
        aggregation,
    )
    row = DifferentiationBenchmarkDataRow(;
        backend = backend,
        scenario = scenario,
        operator = Symbol(operator),
        prepared = prepared,
        calls = calls,
        samples = length(bench.samples),
        evals = Int(bench.samples[1].evals),
        time = aggregation(getfield.(bench.samples, :time)),
        allocs = aggregation(getfield.(bench.samples, :allocs)),
        bytes = aggregation(getfield.(bench.samples, :bytes)),
        gc_fraction = aggregation(getfield.(bench.samples, :gc_fraction)),
        compile_fraction = aggregation(getfield.(bench.samples, :compile_fraction)),
    )
    return push!(data.rows, row)
end

include("benchmark_eval.jl")

end
