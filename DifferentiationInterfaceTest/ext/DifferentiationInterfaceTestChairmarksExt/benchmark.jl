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
        s::Scenario,
        prepared::Union{Nothing, Bool},
        bench::Benchmark,
        calls::Integer,
        aggregation,
    )
    (; samples) = bench
    row = DifferentiationBenchmarkDataRow(;
        backend = backend,
        scenario = s,
        operator = Symbol(s.operator),
        prepared = prepared,
        calls = calls,
        samples = length(bench.samples),
        evals = Int(bench.samples[1].evals),
        time = aggregation(getfield.(samples, :time)),
        allocs = aggregation(getfield.(samples, :allocs)),
        bytes = aggregation(getfield.(samples, :bytes)),
        gc_fraction = aggregation(getfield.(samples, :gc_fraction)),
        compile_fraction = aggregation(getfield.(samples, :compile_fraction)),
    )
    return push!(data.rows, row)
end

function DIT.run_benchmark!(
        data::DifferentiationBenchmark,
        backend::AbstractADType,
        scenario::Scenario;
        logging::Bool,
        subset::Symbol,
        count_calls::Bool,
        benchmark_test::Bool,
        benchmark_seconds::Real,
        benchmark_aggregation,
    )
    @assert subset in (:full, :prepared)

    bench_success = true
    bench_result = try
        benchmark_aux(backend, scenario; subset, s = benchmark_seconds)
    catch exception
        bench_success = false
        logging && @warn "Error during benchmarking" backend scenario exception
        BenchmarkResult()
    end
    benchmark_test && @test bench_success

    if count_calls
        count_success = true
        calls_result = try
            calls_aux(backend, scenario; subset, s = nothing)
        catch exception
            count_success = false
            logging && @warn "Error during call counting" backend scenario exception
            CallsResult()
        end
        benchmark_test && @test count_success
    else
        calls_result = CallsResult()
    end

    record!(
        data;
        backend,
        scenario,
        operator = valop_string,
        prepared = true,
        bench = bench_result.prepared_valop,
        calls = calls_result.prepared_valop,
        aggregation = benchmark_aggregation,
    )
    record!(
        data;
        backend,
        scenario,
        operator = op_string,
        prepared = true,
        bench = bench_result.prepared_op,
        calls = calls_result.prepared_op,
        aggregation = benchmark_aggregation,
    )
    return if subset == :full
        record!(
            data;
            backend,
            scenario,
            operator = prep_string,
            prepared = nothing,
            bench = bench_result.preparation,
            calls = calls_result.preparation,
            aggregation = benchmark_aggregation,
        )
        record!(
            data;
            backend,
            scenario,
            operator = valop_string,
            prepared = false,
            bench = bench_result.unprepared_valop,
            calls = calls_result.unprepared_valop,
            aggregation = benchmark_aggregation,
        )
        record!(
            data;
            backend,
            scenario,
            operator = op_string,
            prepared = false,
            bench = bench_result.unprepared_op,
            calls = calls_result.unprepared_op,
            aggregation = benchmark_aggregation,
        )
    end
end

@eval function benchmark_aux(
        ba::AbstractADType, scen::$S1out; subset::Symbol, s
    )
    (; f, x, contexts, prep_args) = deepcopy(scen)
    prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
    prepared_valop = @be prep $val_and_op(f, _, ba, x, contexts...) seconds = s
    prepared_op = @be prep $op(f, _, ba, x, contexts...) seconds = s
    if subset == :full
        preparation = @be $prep_op(f, ba, prep_args.x, prep_args.contexts...) seconds =
            s
        unprepared_valop = @be $val_and_op(f, ba, x, contexts...) seconds = s
        unprepared_op = @be $op(f, ba, x, contexts...) seconds = s
        return BenchmarkResult(;
            prepared_valop,
            prepared_op,
            preparation,
            unprepared_valop,
            unprepared_op,
        )
    else
        return BenchmarkResult(; prepared_valop, prepared_op)
    end
end

@eval function calls_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
    (; f, x, contexts, prep_args) = deepcopy(scen)
    cc = CallCounter(f)
    prep = $prep_op(cc, ba, prep_args.x, prep_args.contexts...)
    preparation = reset_count!(cc)
    $val_and_op(cc, prep, ba, x, contexts...)
    prepared_valop = reset_count!(cc)
    $op(cc, prep, ba, x, contexts...)
    prepared_op = reset_count!(cc)
    $val_and_op(cc, ba, x, contexts...)
    unprepared_valop = reset_count!(cc)
    $op(cc, ba, x, contexts...)
    unprepared_op = reset_count!(cc)
    return CallsResult(;
        prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
    )
end
