"""
$(TYPEDSIGNATURES)

Apply a list of `backends` on a list of `scenarios`, running a variety of different tests and/or benchmarks.

# Return

This function always creates and runs a `@testset`, though its contents may vary.
    
- if `benchmark == :none`, it returns `nothing`.
- if `benchmark != :none`, it returns a `DataFrame` of benchmark results, whose columns correspond to the fields of [`DifferentiationBenchmarkDataRow`](@ref).

# Positional arguments

- `backends::Vector{<:AbstractADType}`: the backends to test
- `scenarios::Vector{<:Scenario}`: the scenarios on which to test them (defaults to the output of [`default_scenarios()`](@ref))

# Keyword arguments

**Test categories:**

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario
- `type_stability=:none`: whether (and how) to check type stability of operators with JET.jl.
- `allocations=:none`: whether (and how) to check allocations inside operators with AllocCheck.jl
- `benchmark=:none`: whether (and how) to benchmark operators with Chairmarks.jl

For `type_stability`, `allocations` and `benchmark`, the possible values are `:none`, `:prepared` or `:full`.
Each setting tests/benchmarks a different subset of calls:

| kwarg | prepared operator | unprepared operator | preparation |
|---|---|---|---|
| `:none` | no | no | no |
| `:prepared` | yes | no | no |
| `:full` | yes | yes | yes |

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

**Type stability options:**

- `ignored_modules=nothing`: list of modules that JET.jl should ignore
- `function_filter`: filter for functions that JET.jl should ignore (with a reasonable default)

**Benchmark options:**

- `count_calls=true`: whether to also count function calls during benchmarking
- `benchmark_test=true`: whether to include tests which succeed iff benchmark doesn't error
- `benchmark_seconds=1`: how long to run each benchmark for
- `benchmark_aggregation=minimum`: function used to aggregate sample measurements
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:Scenario}=default_scenarios(),
    otherscenarios=deepcopy(scenarios);
    # test categories
    correctness::Bool=true,
    type_stability::Symbol=:none,
    allocations::Symbol=:none,
    benchmark::Symbol=:none,
    # misc options
    excluded::Vector{Symbol}=Symbol[],
    detailed::Bool=false,
    logging::Bool=false,
    # correctness options
    isapprox=isapprox,
    atol::Real=0,
    rtol::Real=1e-3,
    scenario_intact::Bool=true,
    sparsity::Bool=false,
    # type stability options
    ignored_modules=nothing,
    function_filter=if VERSION >= v"1.11"
        @nospecialize(f) -> true
    else
        @nospecialize(f) -> f != Base.mapreduce_empty  # fix for `mapreduce` in jacobian and hessian
    end,
    # allocs options
    skip_allocations::Bool=false,  # private, only for code coverage
    # benchmark options
    count_calls::Bool=true,
    benchmark_test::Bool=true,
    benchmark_seconds::Real=1,
    benchmark_aggregation=minimum,
)
    @assert type_stability in (:none, :prepared, :full)
    @assert allocations in (:none, :prepared, :full)
    @assert benchmark in (:none, :prepared, :full)

    scenario_inds = filter(l -> !(operator(scenarios[l]) in excluded), eachindex(scenarios))
    scenario_inds = sort(
        scenario_inds; by=l -> (operator(scenarios[l]), string(scenarios[l].f))
    )

    title_additions =
        (correctness ? " + correctness" : "") *
        ((type_stability != :none) ? " + type stability" : "") *
        ((benchmark != :none) ? " + benchmarks" : "")
    title = "Testing" * title_additions[3:end]

    benchmark_data = DifferentiationBenchmarkDataRow[]

    prog = ProgressUnknown(; desc="$title", spinner=true, enabled=logging)

    @testset verbose = true "$title" begin
        @testset verbose = detailed "$backend" for (i, backend) in enumerate(backends)
            filtered_scenario_inds = filter(
                l -> compatible(backend, scenarios[l]), scenario_inds
            )
            grouped_scenario_inds = group_by_operator(filtered_scenario_inds, scenarios)
            @testset verbose = detailed "$op" for (j, (op, op_group)) in
                                                  enumerate(pairs(grouped_scenario_inds))
                @testset "$(scenarios[l])" for (k, l) in enumerate(op_group)
                    scen = scenarios[l]
                    otherscen = otherscenarios[l]
                    next!(
                        prog;
                        showvalues=[
                            (:backend, "$backend - $i/$(length(backends))"),
                            (:scenario_type, "$op - $j/$(length(grouped_scenario_inds))"),
                            (:scenario, "$k/$(length(op_group))"),
                            (:operator_place, operator_place(scen)),
                            (:function_place, function_place(scen)),
                            (:function, scen.f),
                            (:input_type, typeof(scen.x)),
                            (:input_size, mysize(scen.x)),
                            (:output_type, typeof(scen.y)),
                            (:output_size, mysize(scen.y)),
                            (
                                :nb_tangents,
                                scen.tang isa NTuple ? length(scen.tang) : nothing,
                            ),
                            (:nb_contexts, length(scen.contexts)),
                        ],
                    )
                    adapted_backend = adapt_batchsize(backend, scen)
                    correctness && @testset "Correctness" begin
                        test_correctness(
                            adapted_backend,
                            scen,
                            otherscen;
                            isapprox,
                            atol,
                            rtol,
                            scenario_intact,
                            sparsity,
                        )
                    end
                    yield()
                    (type_stability != :none) && @testset "Type stability" begin
                        test_jet(
                            adapted_backend,
                            scen;
                            subset=type_stability,
                            ignored_modules,
                            function_filter,
                        )
                    end
                    yield()
                    (allocations != :none) && @testset "Allocations" begin
                        test_alloccheck(
                            adapted_backend,
                            scen;
                            subset=allocations,
                            skip=skip_allocations,
                        )
                    end
                    yield()
                    (benchmark != :none) && @testset "Benchmark" begin
                        run_benchmark!(
                            benchmark_data,
                            adapted_backend,
                            scen;
                            logging,
                            subset=benchmark,
                            count_calls,
                            benchmark_test,
                            benchmark_seconds,
                            benchmark_aggregation,
                        )
                    end
                    yield()
                end
            end
        end
    end
    if benchmark != :none
        return DataFrame(benchmark_data)
    else
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_differentiation(backend::AbstractADType, args...; kwargs...)
    return test_differentiation([backend], args...; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Shortcut for [`test_differentiation`](@ref) with only benchmarks and no correctness or type stability checks.

Specifying the set of scenarios is mandatory for this function.
"""
function benchmark_differentiation(
    backends,
    scenarios::Vector{<:Scenario};
    benchmark::Symbol=:prepared,
    excluded::Vector{Symbol}=Symbol[],
    logging::Bool=false,
    count_calls::Bool=true,
    benchmark_test::Bool=true,
    benchmark_seconds::Real=1,
    benchmark_aggregation=minimum,
)
    return test_differentiation(
        backends,
        scenarios;
        correctness=false,
        type_stability=:none,
        allocations=:none,
        benchmark,
        logging,
        excluded,
        count_calls,
        benchmark_test,
        benchmark_seconds,
        benchmark_aggregation,
    )
end
