struct CallCounter{F}
    f::F
    count::Base.RefValue{Int}
end

CallCounter(f::F) where {F} = CallCounter{F}(f, Ref(0))

function (cc::CallCounter)(x)
    cc.count[] += 1
    return cc.f(x)
end

function (cc::CallCounter)(y, x)
    cc.count[] += 1
    return cc.f(y, x)
end

function reset_count!(cc::CallCounter)
    count = cc.count[]
    cc.count[] = 0
    return count
end

function failed_bench()
    evals = 0
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

function failed_benchs(k::Integer)
    return ntuple(i -> failed_bench(), k)
end

"""
    DifferentiationBenchmarkDataRow

Ad-hoc storage type for differentiation benchmarking results.

If you have a vector `rows::Vector{DifferentiationBenchmarkDataRow}`, you can turn it into a `DataFrame` as follows:

```julia
using DataFrames

df = DataFrame(rows)
```

The resulting `DataFrame` will have one column for each of the following fields.

#  Fields

$(TYPEDFIELDS)

See the documentation of [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl) for more details on the measurement fields.
"""
Base.@kwdef struct DifferentiationBenchmarkDataRow
    "backend used for benchmarking"
    backend::AbstractADType
    "scenario used for benchmarking"
    scenario::AbstractScenario
    "differentiation operator used for benchmarking, e.g. `:gradient` or `:hessian`"
    operator::Symbol
    "number of calls to the differentiated function for one call to the operator"
    calls::Int
    "number of benchmarking samples taken"
    samples::Int
    "number of evaluations used for averaging in each sample"
    evals::Int
    "minimum runtime over all samples, in seconds"
    time::Float64
    "minimum number of allocations over all samples"
    allocs::Float64
    "minimum memory allocated over all samples, in bytes"
    bytes::Float64
    "minimum fraction of time spent in garbage collection over all samples, between 0.0 and 1.0"
    gc_fraction::Float64
    "minimum fraction of time spent compiling over all samples, between 0.0 and 1.0"
    compile_fraction::Float64
end

function record!(
    data::Vector{DifferentiationBenchmarkDataRow},
    backend::AbstractADType,
    scenario::AbstractScenario,
    operator::Symbol,
    bench::Benchmark,
    calls::Integer,
)
    bench_min = minimum(bench)
    row = DifferentiationBenchmarkDataRow(;
        backend=backend,
        scenario=scenario,
        operator=operator,
        calls=calls,
        samples=length(bench.samples),
        evals=Int(bench_min.evals),
        time=bench_min.time,
        allocs=bench_min.allocs,
        bytes=bench_min.bytes,
        gc_fraction=bench_min.gc_fraction,
        compile_fraction=bench_min.compile_fraction,
    )
    return push!(data, row)
end

## Pushforward

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y, dx) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f, ba, x, dx)
        bench0 = @be prepare_pushforward(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) value_and_pushforward(f, ba, x, dx, _) evals = 1
        bench2 = @be deepcopy(extras) pushforward(f, ba, x, dx, _) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_pushforward(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        value_and_pushforward(cc, ba, x, dx, extras)
        calls1 = reset_count!(cc)
        pushforward(cc, ba, x, dx, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward, bench1, calls1)
    record!(data, ba, scen, :pushforward, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y, dx) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f, ba, x, dx)
        bench0 = @be prepare_pushforward(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be (dy=mysimilar(y), ext=deepcopy(extras)) value_and_pushforward!(
            f, _.dy, ba, x, dx, _.ext
        ) evals = 1
        bench2 = @be (dy=mysimilar(y), ext=deepcopy(extras)) pushforward!(
            f, _.dy, ba, x, dx, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_pushforward(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        value_and_pushforward!(cc, mysimilar(y), ba, x, dx, extras)
        calls1 = reset_count!(cc)
        pushforward!(cc, mysimilar(y), ba, x, dx, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward!, bench1, calls1)
    record!(data, ba, scen, :pushforward!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{2,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y, dx) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f!, mysimilar(y), ba, x, dx)
        bench0 = @be mysimilar(y) prepare_pushforward(f!, _, ba, x, dx) samples = 1 evals =
            1
        bench1 = @be (y=mysimilar(y), ext=deepcopy(extras)) value_and_pushforward(
            f!, _.y, ba, x, dx, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), ext=deepcopy(extras)) pushforward(
            f!, _.y, ba, x, dx, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pushforward(cc!, mysimilar(y), ba, x, dx)
        calls0 = reset_count!(cc!)
        value_and_pushforward(cc!, mysimilar(y), ba, x, dx, extras)
        calls1 = reset_count!(cc!)
        pushforward(cc!, mysimilar(y), ba, x, dx, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward, bench1, calls1)
    record!(data, ba, scen, :pushforward, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PushforwardScenario{2,:inplace};
    logging::Bool,
)
    @compat (; f, x, y, dx) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pushforward(f!, y, ba, x, dx)
        bench0 = @be mysimilar(y) prepare_pushforward(f!, _, ba, x, dx) evals = 1 samples =
            1
        bench1 = @be (y=mysimilar(y), dy=mysimilar(y), ext=deepcopy(extras)) value_and_pushforward!(
            f!, _.y, _.dy, ba, x, dx, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), dy=mysimilar(y), ext=deepcopy(extras)) pushforward!(
            f!, _.y, _.dy, ba, x, dx, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pushforward(cc!, mysimilar(y), ba, x, dx)
        calls0 = reset_count!(cc!)
        value_and_pushforward!(cc!, mysimilar(y), mysimilar(y), ba, x, dx, extras)
        calls1 = reset_count!(cc!)
        pushforward!(cc!, mysimilar(y), mysimilar(y), ba, x, dx, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pushforward, bench0, calls0)
    record!(data, ba, scen, :value_and_pushforward!, bench1, calls1)
    record!(data, ba, scen, :pushforward!, bench2, calls2)
    return nothing
end

## Pullback

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PullbackScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y, dy) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pullback(f, ba, x, dy)
        bench0 = @be prepare_pullback(f, ba, x, dy) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) value_and_pullback(f, ba, x, dy, _)
        bench2 = @be deepcopy(extras) pullback(f, ba, x, dy, _)
        # count
        cc = CallCounter(f)
        extras = prepare_pullback(cc, ba, x, dy)
        calls0 = reset_count!(cc)
        value_and_pullback(cc, ba, x, dy, extras)
        calls1 = reset_count!(cc)
        pullback(cc, ba, x, dy, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback, bench1, calls1)
    record!(data, ba, scen, :pullback, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PullbackScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y, dy) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pullback(f, ba, x, dy)
        bench0 = @be prepare_pullback(f, ba, x, dy) samples = 1 evals = 1
        bench1 = @be (dx=mysimilar(x), ext=deepcopy(extras)) value_and_pullback!(
            f, _.dx, ba, x, dy, _.ext
        ) evals = 1
        bench2 = @be (dx=mysimilar(x), ext=deepcopy(extras)) pullback!(
            f, _.dx, ba, x, dy, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_pullback(cc, ba, x, dy)
        calls0 = reset_count!(cc)
        value_and_pullback!(cc, mysimilar(x), ba, x, dy, extras)
        calls1 = reset_count!(cc)
        pullback!(cc, mysimilar(x), ba, x, dy, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback!, bench1, calls1)
    record!(data, ba, scen, :pullback!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PullbackScenario{2,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y, dy) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pullback(f!, mysimilar(y), ba, x, dy)
        bench0 = @be mysimilar(y) prepare_pullback(f!, _, ba, x, dy) samples = 1 evals =
            1
        bench1 = @be (y=mysimilar(y), ext=deepcopy(extras)) value_and_pullback(
            f!, _.y, ba, x, dy, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), ext=deepcopy(extras)) pullback(
            f!, _.y, ba, x, dy, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pullback(cc!, mysimilar(y), ba, x, dy)
        calls0 = reset_count!(cc!)
        value_and_pullback(cc!, mysimilar(y), ba, x, dy, extras)
        calls1 = reset_count!(cc!)
        pullback(cc!, mysimilar(y), ba, x, dy, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback, bench1, calls1)
    record!(data, ba, scen, :pullback, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::PullbackScenario{2,:inplace};
    logging::Bool,
)
    @compat (; f, x, y, dy) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_pullback(f!, mysimilar(y), ba, x, dy)
        bench0 = @be mysimilar(y) prepare_pullback(f!, _, ba, x, dy) samples = 1 evals =
            1
        bench1 = @be (y=mysimilar(y), dx=mysimilar(x), ext=deepcopy(extras)) value_and_pullback!(
            f!, _.y, _.dx, ba, x, dy, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), dx=mysimilar(x), ext=deepcopy(extras)) pullback!(
            f!, _.y, _.dx, ba, x, dy, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_pullback(cc!, mysimilar(y), ba, x, dy)
        calls0 = reset_count!(cc!)
        value_and_pullback!(cc!, mysimilar(y), mysimilar(x), ba, x, dy, extras)
        calls1 = reset_count!(cc!)
        pullback!(cc!, mysimilar(y), mysimilar(x), ba, x, dy, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_pullback, bench0, calls0)
    record!(data, ba, scen, :value_and_pullback!, bench1, calls1)
    record!(data, ba, scen, :pullback!, bench2, calls2)
    return nothing
end

## Derivative

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f, ba, x)
        bench0 = @be prepare_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) value_and_derivative(f, ba, x, _)
        bench2 = @be deepcopy(extras) derivative(f, ba, x, _)
        # count
        cc = CallCounter(f)
        extras = prepare_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_derivative(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        derivative(cc, ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative, bench1, calls1)
    record!(data, ba, scen, :derivative, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f, ba, x)
        bench0 = @be prepare_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be (der=mysimilar(y), ext=deepcopy(extras)) value_and_derivative!(
            f, _.der, ba, x, _.ext
        ) evals = 1
        bench2 = @be (der=mysimilar(y), ext=deepcopy(extras)) derivative!(
            f, _.der, ba, x, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_derivative!(cc, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc)
        derivative!(cc, mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative!, bench1, calls1)
    record!(data, ba, scen, :derivative!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{2,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_derivative(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be (y=mysimilar(y), ext=deepcopy(extras)) value_and_derivative(
            f!, _.y, ba, x, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), ext=deepcopy(extras)) derivative(
            f!, _.y, ba, x, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_derivative(cc!, mysimilar(y), ba, x)
        calls0 = reset_count!(cc!)
        value_and_derivative(cc!, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc!)
        derivative(cc!, mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative, bench1, calls1)
    record!(data, ba, scen, :derivative, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::DerivativeScenario{2,:inplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_derivative(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_derivative(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be (y=mysimilar(y), der=mysimilar(y), ext=deepcopy(extras)) value_and_derivative!(
            f!, _.y, _.der, ba, x, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), der=mysimilar(y), ext=deepcopy(extras)) derivative!(
            f!, _.y, _.der, ba, x, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_derivative(cc!, mysimilar(y), ba, x)
        calls0 = reset_count!(cc!)
        value_and_derivative!(cc!, mysimilar(y), mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc!)
        derivative!(cc!, mysimilar(y), mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_derivative, bench0, calls0)
    record!(data, ba, scen, :value_and_derivative!, bench1, calls1)
    record!(data, ba, scen, :derivative!, bench2, calls2)
    return nothing
end

## Gradient

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::GradientScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_gradient(f, ba, x)
        bench0 = @be prepare_gradient(f, ba, x) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) value_and_gradient(f, ba, x, _)
        bench2 = @be deepcopy(extras) gradient(f, ba, x, _)
        # count
        cc = CallCounter(f)
        extras = prepare_gradient(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_gradient(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        gradient(cc, ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_gradient, bench0, calls0)
    record!(data, ba, scen, :value_and_gradient, bench1, calls1)
    record!(data, ba, scen, :gradient, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::GradientScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_gradient(f, ba, x)
        bench0 = @be prepare_gradient(f, ba, x) samples = 1 evals = 1
        bench1 = @be (grad=mysimilar(x), ext=deepcopy(extras)) value_and_gradient!(
            f, _.grad, ba, x, _.ext
        ) evals = 1
        bench2 = @be (grad=mysimilar(x), ext=deepcopy(extras)) gradient!(
            f, _.grad, ba, x, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_gradient(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_gradient!(cc, mysimilar(x), ba, x, extras)
        calls1 = reset_count!(cc)
        gradient!(cc, mysimilar(x), ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_gradient, bench0, calls0)
    record!(data, ba, scen, :value_and_gradient!, bench1, calls1)
    record!(data, ba, scen, :gradient!, bench2, calls2)
    return nothing
end

## Jacobian

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::JacobianScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_jacobian(f, ba, x)
        bench0 = @be prepare_jacobian(f, ba, x) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) value_and_jacobian(f, ba, x, _)
        bench2 = @be deepcopy(extras) jacobian(f, ba, x, _)
        # count
        cc = CallCounter(f)
        extras = prepare_jacobian(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_jacobian(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        jacobian(cc, ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian, bench1, calls1)
    record!(data, ba, scen, :jacobian, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::JacobianScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        jac_template = mysimilar(jacobian(f, ba, x))
        # benchmark
        extras = prepare_jacobian(f, ba, x)
        bench0 = @be prepare_jacobian(f, ba, x) samples = 1 evals = 1
        bench1 = @be (jac=mysimilar(jac_template), ext=deepcopy(extras)) value_and_jacobian!(
            f, _.jac, ba, x, _.ext
        ) evals = 1
        bench2 = @be (jac=mysimilar(jac_template), ext=deepcopy(extras)) jacobian!(
            f, _.jac, ba, x, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_jacobian(cc, ba, x)
        calls0 = reset_count!(cc)
        value_and_jacobian!(cc, mysimilar(jac_template), ba, x, extras)
        calls1 = reset_count!(cc)
        jacobian!(cc, mysimilar(jac_template), ba, x, extras)
        calls2 = reset_count!(cc)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian!, bench1, calls1)
    record!(data, ba, scen, :jacobian!, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::JacobianScenario{2,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        # benchmark
        extras = prepare_jacobian(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_jacobian(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be (y=mysimilar(y), ext=deepcopy(extras)) value_and_jacobian(
            f!, _.y, ba, x, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), ext=deepcopy(extras)) jacobian(f!, _.y, ba, x, _.ext) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_jacobian(cc!, mysimilar(y), ba, x)
        calls0 = reset_count!(cc!)
        value_and_jacobian(cc!, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc!)
        jacobian(cc!, mysimilar(y), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian, bench1, calls1)
    record!(data, ba, scen, :jacobian, bench2, calls2)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::JacobianScenario{2,:inplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
        jac_template = mysimilar(jacobian(f!, mysimilar(y), ba, x))
        # benchmark
        extras = prepare_jacobian(f!, mysimilar(y), ba, x)
        bench0 = @be mysimilar(y) prepare_jacobian(f!, _, ba, x) samples = 1 evals = 1
        bench1 = @be (y=mysimilar(y), jac=mysimilar(jac_template), ext=deepcopy(extras)) value_and_jacobian!(
            f!, _.y, _.jac, ba, x, _.ext
        ) evals = 1
        bench2 = @be (y=mysimilar(y), jac=mysimilar(jac_template), ext=deepcopy(extras)) jacobian!(
            f!, _.y, _.jac, ba, x, _.ext
        ) evals = 1
        # count
        cc! = CallCounter(f!)
        extras = prepare_jacobian(cc!, y, ba, x)
        calls0 = reset_count!(cc!)
        value_and_jacobian!(cc!, mysimilar(y), mysimilar(jac_template), ba, x, extras)
        calls1 = reset_count!(cc!)
        jacobian!(cc!, mysimilar(y), mysimilar(jac_template), ba, x, extras)
        calls2 = reset_count!(cc!)
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1, bench2 = failed_benchs(3)
        calls0, calls1, calls2 = -1, -1, -1
        (; bench0, bench1, bench2, calls0, calls1, calls2)
    end
    # record
    record!(data, ba, scen, :prepare_jacobian, bench0, calls0)
    record!(data, ba, scen, :value_and_jacobian!, bench1, calls1)
    record!(data, ba, scen, :jacobian!, bench2, calls2)
    return nothing
end

## Second derivative

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_second_derivative(f, ba, x)
        bench0 = @be prepare_second_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) second_derivative(f, ba, x, _)
        # count
        cc = CallCounter(f)
        extras = prepare_second_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        second_derivative(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_second_derivative, bench0, calls0)
    record!(data, ba, scen, :second_derivative, bench1, calls1)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_second_derivative(f, ba, x)
        bench0 = @be prepare_second_derivative(f, ba, x) samples = 1 evals = 1
        bench1 = @be (der=mysimilar(y), ext=deepcopy(extras)) second_derivative!(
            f, _.der, ba, x, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_second_derivative(cc, ba, x)
        calls0 = reset_count!(cc)
        second_derivative!(cc, mysimilar(y), ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_second_derivative, bench0, calls0)
    record!(data, ba, scen, :second_derivative!, bench1, calls1)
    return nothing
end

## Hessian-vector product

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::HVPScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y, dx) = deepcopy(scen)
    @compat (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_hvp(f, ba, x, dx)
        bench0 = @be prepare_hvp(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) hvp(f, ba, x, dx, _)
        # count
        cc = CallCounter(f)
        extras = prepare_hvp(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        hvp(cc, ba, x, dx, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hvp, bench0, calls0)
    record!(data, ba, scen, :hvp, bench1, calls1)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::HVPScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y, dx) = deepcopy(scen)
    @compat (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_hvp(f, ba, x, dx)
        bench0 = @be prepare_hvp(f, ba, x, dx) samples = 1 evals = 1
        bench1 = @be (p=mysimilar(x), ext=deepcopy(extras)) hvp!(f, _.p, ba, x, dx, _.ext) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_hvp(cc, ba, x, dx)
        calls0 = reset_count!(cc)
        hvp!(cc, mysimilar(x), ba, x, dx, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hvp, bench0, calls0)
    record!(data, ba, scen, :hvp!, bench1, calls1)
    return nothing
end

## Hessian

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::HessianScenario{1,:outofplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, calls0, calls1) = try
        # benchmark
        extras = prepare_hessian(f, ba, x)
        bench0 = @be prepare_hessian(f, ba, x) samples = 1 evals = 1
        bench1 = @be deepcopy(extras) hessian(f, ba, x, _)
        # count
        cc = CallCounter(f)
        extras = prepare_hessian(cc, ba, x)
        calls0 = reset_count!(cc)
        hessian(cc, ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hessian, bench0, calls0)
    record!(data, ba, scen, :hessian, bench1, calls1)
    return nothing
end

function run_benchmark!(
    data::Vector{DifferentiationBenchmarkDataRow},
    ba::AbstractADType,
    scen::HessianScenario{1,:inplace};
    logging::Bool,
)
    @compat (; f, x, y) = deepcopy(scen)
    @compat (; bench0, bench1, calls0, calls1) = try
        hess_template = Matrix{typeof(y)}(undef, length(x), length(x))
        # benchmark
        extras = prepare_hessian(f, ba, x)
        bench0 = @be prepare_hessian(f, ba, x) samples = 1 evals = 1
        bench1 = @be (hess=mysimilar(hess_template), ext=deepcopy(extras)) hessian!(
            f, _.hess, ba, x, _.ext
        ) evals = 1
        # count
        cc = CallCounter(f)
        extras = prepare_hessian(cc, ba, x)
        calls0 = reset_count!(cc)
        hessian!(cc, mysimilar(hess_template), ba, x, extras)
        calls1 = reset_count!(cc)
        (; bench0, bench1, calls0, calls1)
    catch e
        logging && @warn "Error during benchmarking" ba scen e
        bench0, bench1 = failed_benchs(2)
        calls0, calls1 = -1, -1
        (; bench0, bench1, calls0, calls1)
    end
    # record
    record!(data, ba, scen, :prepare_hessian, bench0, calls0)
    record!(data, ba, scen, :hessian!, bench1, calls1)
    return nothing
end
