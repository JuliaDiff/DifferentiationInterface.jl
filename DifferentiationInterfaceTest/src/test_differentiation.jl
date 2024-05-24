"""
$(TYPEDSIGNATURES)

Cross-test a list of `backends` on a list of `scenarios`, running a variety of different tests.

# Default arguments

- `scenarios::Vector{<:AbstractScenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario
- `type_stability=false`: whether to check type stability with JET.jl (thanks to `JET.@test_opt`)
- `sparsity`: whether to check sparsity of the jacobian / hessian
- `ref_backend`: if not `nothing`, an `ADTypes.AbstractADType` object to use instead of the scenario-specific reference to provide true values
- `detailed=false`: whether to print a detailed or condensed test log

Filtering:

- `input_type=Any`, `output_type=Any`: restrict scenario inputs / outputs to subtypes of this
- `first_order=true`, `second_order=true`: include first order / second order operators
- `onearg=true`, `twoarg=true`: include one-argument / two-argument functions
- `inplace=true`, `outofplace=true`: include in-place / out-of-place operators

Options:

- `logging=false`: whether to log progress
- `isapprox=isapprox`: function used to compare objects, with the standard signature `isapprox(x, y; atol, rtol)`
- `atol=0`: absolute precision for correctness testing (when comparing to the reference outputs)
- `rtol=1e-3`: relative precision for correctness testing (when comparing to the reference outputs)
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:AbstractScenario}=default_scenarios();
    # testing
    correctness::Bool=true,
    type_stability::Bool=false,
    call_count::Bool=false,
    sparsity::Bool=false,
    ref_backend=nothing,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    first_order=true,
    second_order=true,
    onearg=true,
    twoarg=true,
    inplace=true,
    outofplace=true,
    excluded=[],
    # options
    logging=false,
    isapprox=isapprox,
    atol=0,
    rtol=1e-3,
)
    scenarios = filter_scenarios(
        scenarios;
        first_order,
        second_order,
        input_type,
        output_type,
        onearg,
        twoarg,
        inplace,
        outofplace,
        excluded,
    )

    title_additions =
        (correctness != false ? " + correctness" : "") *
        (call_count ? " + calls" : "") *
        (type_stability ? " + types" : "") *
        (sparsity ? " + sparsity" : "")
    title = "Testing" * title_additions[3:end]

    prog = ProgressUnknown(; desc="$title", spinner=true, enabled=logging)

    @testset verbose = true "$title" begin
        @testset verbose = detailed "$(backend_str(backend))" for (i, backend) in
                                                                  enumerate(backends)
            filtered_scenarios = filter(s -> compatible(backend, s), scenarios)
            grouped_scenarios = group_by_scen_type(filtered_scenarios)
            @testset verbose = detailed "$st" for (j, (st, st_group)) in
                                                  enumerate(pairs(grouped_scenarios))
                @testset "$scen" for (k, scen) in enumerate(st_group)
                    next!(
                        prog;
                        showvalues=[
                            (:backend, "$(backend_str(backend)) - $i/$(length(backends))"),
                            (:scenario_type, "$st - $j/$(length(grouped_scenarios))"),
                            (:scenario, "$k/$(length(st_group))"),
                            (:arguments, nb_args(scen)),
                            (:place, operator_place(scen)),
                            (:function, scen.f),
                            (:input_type, typeof(scen.x)),
                            (:input_size, size(scen.x)),
                            (:output_type, typeof(scen.y)),
                            (:output_size, size(scen.y)),
                        ],
                    )
                    correctness && @testset "Correctness" begin
                        test_correctness(backend, scen; isapprox, atol, rtol, ref_backend)
                    end
                    type_stability && @testset "Type stability" begin
                        @static if VERSION >= v"1.7"
                            test_jet(backend, scen; ref_backend)
                        end
                    end
                    sparsity && @testset "Sparsity" begin
                        test_sparsity(backend, scen; ref_backend)
                    end
                end
            end
        end
    end
    return nothing
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

Benchmark a list of `backends` for a list of `operators` on a list of `scenarios`.

The object returned is a `Vector` of [`DifferentiationBenchmarkDataRow`](@ref).

The keyword arguments available here have the same meaning as those in [`test_differentiation`](@ref).
"""
function benchmark_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:AbstractScenario};
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    first_order=true,
    second_order=true,
    onearg=true,
    twoarg=true,
    inplace=true,
    outofplace=true,
    excluded=[],
    # options
    logging=false,
)
    scenarios = filter_scenarios(
        scenarios;
        first_order,
        second_order,
        input_type,
        output_type,
        onearg,
        twoarg,
        inplace,
        outofplace,
        excluded,
    )

    benchmark_data = DifferentiationBenchmarkDataRow[]
    prog = ProgressUnknown(; desc="Benchmarking", spinner=true, enabled=logging)
    for (i, backend) in enumerate(backends)
        filtered_scenarios = filter(s -> compatible(backend, s), scenarios)
        grouped_scenarios = group_by_scen_type(filtered_scenarios)
        for (j, (st, st_group)) in enumerate(pairs(grouped_scenarios))
            for (k, scen) in enumerate(st_group)
                next!(
                    prog;
                    showvalues=[
                        (:backend, "$(backend_str(backend)) - $i/$(length(backends))"),
                        (:scenario_type, "$st - $j/$(length(grouped_scenarios))"),
                        (:scenario, "$k/$(length(st_group))"),
                        (:arguments, nb_args(scen)),
                        (:place, operator_place(scen)),
                        (:function, scen.f),
                        (:input_type, typeof(scen.x)),
                        (:input_size, size(scen.x)),
                        (:output_type, typeof(scen.y)),
                        (:output_size, size(scen.y)),
                    ],
                )
                run_benchmark!(benchmark_data, backend, scen; logging)
            end
        end
    end
    return benchmark_data
end
