@enum TestSubset TEST_FULL TEST_PREPARED TEST_NONE

"""
$(TYPEDSIGNATURES)

Apply a list of `backends` on a list of `scenarios`, running a variety of different tests and/or benchmarks.

# Return

This function always creates and runs a `@testset`, though its contents may vary.

  - if `benchmark == :none`, it returns `nothing`.
  - if `benchmark != :none`, it returns a table of benchmark results, compatible with the [Tables.jl](https://github.com/JuliaData/Tables.jl) interface.

# Positional arguments

  - `backends::Vector{<:AbstractADType}`: the backends to test
  - `scenarios::Vector{<:Scenario}`: the scenarios on which to test these backends. Defaults to a standard set of first- and second-order scenarios, whose contents are not part of the public API and may change without notice.

# Keyword arguments

  - `testset_name=nothing`: how to display the test set

**Test categories:**

  - `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario
  - `type_stability=:none`: whether (and how) to check type stability of operators with JET.jl.
  - `allocations=:none`: whether (and how) to check allocations inside operators with AllocCheck.jl
  - `benchmark=:none`: whether (and how) to benchmark operators with Chairmarks.jl

For `type_stability`, `allocations` and `benchmark`, the possible values are `:none`, `:prepared` or `:full`.
Each setting tests/benchmarks a different subset of calls:

| kwarg       | prepared operator | unprepared operator | preparation |
|:----------- |:----------------- |:------------------- |:----------- |
| `:none`     | no                | no                  | no          |
| `:prepared` | yes               | no                  | no          |
| `:full`     | yes               | yes                 | yes         |

**Misc options:**

  - `excluded::Vector{Symbol}`: list of operators to exclude, such as [`FIRST_ORDER`](@ref) or [`SECOND_ORDER`](@ref)
  - `detailed=false`: whether to create a detailed or condensed testset
  - `logging=false`: whether to log progress

**Correctness options:**

  - `isapprox=isapprox`: function used to compare objects approximately, with the standard signature `isapprox(x, y; atol, rtol)`
  - `atol=0`: absolute precision for correctness testing (when comparing to the reference outputs)
  - `rtol=1e-3`: relative precision for correctness testing (when comparing to the reference outputs)
  - `scenario_intact=true`: whether to check that the scenario remains unchanged after the operators are applied
  - `sparsity=false`: whether to check sparsity patterns for Jacobians / Hessians
  - `reprepare::Bool=true`: whether to modify preparation before testing when the preparation arguments have the wrong size

**Type stability options:**

Type stability checks are implemented in a package extension: please call `import JET` beforehand if you want to use them.

  - `ignored_modules=nothing`: list of modules that JET.jl should ignore
  - `function_filter`: filter for functions that JET.jl should ignore (with a reasonable default)

**Benchmark options:**

Benchmarking is implemented in a package extension: please call `import Chairmarks` beforehand if you want to use it.

  - `count_calls=true`: whether to also count function calls during benchmarking
  - `benchmark_test=true`: whether to include tests which succeed iff benchmark doesn't error
  - `benchmark_seconds=1`: how long to run each benchmark for
  - `benchmark_aggregation=minimum`: function used to aggregate sample measurements

**Batch size options**

  - `adaptive_batchsize=true`: whether to cap the backend's preset batch size (when it exists) to prevent errors on small inputs
"""
function test_differentiation(
        backends::Vector{<:AbstractADType},
        scenarios::Vector{<:TripleScenario};
        testset_name::Union{String, Nothing} = nothing,
        # test categories
        correctness::Bool = true,
        type_stability::TestSubset = TEST_NONE,
        allocations::TestSubset = TEST_NONE,
        benchmark::TestSubset = TEST_NONE,
        # misc options
        excluded::Vector{Symbol} = Symbol[],
        detailed::Bool = false,
        logging::Bool = false,
        # correctness options
        isapprox = isapprox,
        atol::Real = 0,
        rtol::Real = 1.0e-3,
        scenario_intact::Bool = true,
        sparsity::Bool = false,
        reprepare::Bool = true,
        # type stability options
        ignored_modules = nothing,
        function_filter = @nospecialize(f) -> true,
        # allocs options
        skip_allocations::Bool = false,  # private, only for code coverage
        # benchmark options
        count_calls::Bool = true,
        benchmark_test::Bool = true,
        benchmark_seconds::Real = 1,
        benchmark_aggregation = minimum,
        # batch size
        adaptive_batchsize::Bool = true,
    )
    scenarios = filter(s -> !(operator(s) in excluded), scenarios)
    scenarios = sort(scenarios; by = s -> (operator(s), string(s.f)))

    if isnothing(testset_name)
        title_additions =
            (correctness ? " + correctness" : "") *
            ((type_stability != TEST_NONE) ? " + type stability" : "") *
            ((benchmark != TEST_NONE) ? " + benchmarks" : "")
        title = "Testing" * title_additions[3:end]
    else
        title = testset_name
    end

    benchmark_data = DifferentiationBenchmark()

    prog = ProgressUnknown(; desc = "$title", spinner = true, enabled = logging)

    @testset verbose = true "$title" begin
        @testset verbose = detailed "$backend" for (i, backend) in enumerate(backends)
            for (j, s) in enumerate(scenarios)
                @testset verbose = detailed "$s" for (j, s) in enumerate(scenarios)
                    next!(
                        prog;
                        showvalues = [
                            (:backend, "$i/$(length(backends)) - $backend"),
                            (:scenario, "$j/$(length(scenarios)) - $s"),
                        ],
                    )
                    adapted_backend = if adaptive_batchsize
                        adapt_batchsize(backend, s)
                    else
                        backend
                    end
                    if correctness
                        @testset verbose = true "Correctness" begin
                            test_correctness(
                                adapted_backend,
                                s;
                                isapprox,
                                atol,
                                rtol,
                                reprepare,
                            )
                            # test_prep(adapted_backend, scen)
                        end
                    end
                    yield()
                    if type_stability != TEST_NONE
                        @testset verbose = true "Type stability" begin
                            test_type_stability(
                                adapted_backend,
                                s;
                                subset = type_stability,
                                ignored_modules,
                                function_filter,
                            )
                        end
                    end
                    yield()
                    if allocations != TEST_NONE
                        @testset verbose = true "Allocations" begin
                            test_allocations(
                                adapted_backend,
                                s;
                                subset = allocations,
                                skip = skip_allocations,
                            )
                        end
                    end
                    yield()
                    if benchmark != TEST_NONE
                        @testset verbose = true "Benchmark" begin
                            run_benchmark!(
                                benchmark_data,
                                adapted_backend,
                                s;
                                logging,
                                subset = benchmark,
                                count_calls,
                                benchmark_test,
                                benchmark_seconds,
                                benchmark_aggregation,
                            )
                        end
                    end
                    yield()
                end
            end
        end
    end
    return benchmark_data
end
