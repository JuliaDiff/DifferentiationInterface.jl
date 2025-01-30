using Pkg
Pkg.add("ForwardDiff")

using ADTypes: ADTypes
using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using StaticArrays: StaticArrays, @SVector
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

struct MyTag end

backends = [
    AutoForwardDiff(),
    AutoForwardDiff(; chunksize=5),
    AutoForwardDiff(; tag=ForwardDiff.Tag(MyTag(), Float64)),
]

## Dense

test_differentiation(
    backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(),
    default_scenarios(;
        include_normal=false, include_batchified=false, include_cachified=true
    );
    logging=LOGGING,
);

test_differentiation(
    AutoForwardDiff(); correctness=false, type_stability=:prepared, logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(; chunksize=5);
    correctness=false,
    type_stability=:full,
    excluded=[:hessian],
    logging=LOGGING,
);

test_differentiation(
    backends,
    vcat(component_scenarios(), static_scenarios()); # FD accesses individual indices
    excluded=vcat(SECOND_ORDER, [:jacobian]),  # jacobian is super slow for some reason
    logging=LOGGING,
);

## Sparse

test_differentiation(MyAutoSparse(AutoForwardDiff()), default_scenarios(); logging=LOGGING);

test_differentiation(
    MyAutoSparse(AutoForwardDiff()),
    sparse_scenarios(; include_constantified=true);
    sparsity=true,
    logging=LOGGING,
);

## Static

test_differentiation(AutoForwardDiff(), static_scenarios(); logging=LOGGING)

@testset verbose = true "StaticArrays" begin
    @testset "Batch size" begin
        @test DI.pick_batchsize(AutoForwardDiff(), rand(7)) isa DI.BatchSizeSettings{7}
        @test DI.pick_batchsize(AutoForwardDiff(; chunksize=5), rand(7)) ==
            DI.BatchSizeSettings{5}
        @test (@inferred DI.pick_batchsize(AutoForwardDiff(), @SVector(rand(7)))) ==
            DI.BatchSizeSettings{7}
        @test (@inferred DI.pick_batchsize(
            AutoForwardDiff(; chunksize=5), @SVector(rand(7))
        )) isa DI.BatchSizeSettings{5}
    end

    filtered_static_scenarios = filter(static_scenarios(; include_batchified=false)) do scen
        DIT.function_place(scen) == :out && DIT.operator_place(scen) == :out
    end
    data = benchmark_differentiation(
        AutoForwardDiff(),
        filtered_static_scenarios;
        benchmark=:prepared,
        excluded=[:hessian, :pullback],  # TODO: figure this out
        logging=LOGGING,
    )
    @testset "Analyzing benchmark results" begin
        @testset "$(row[:scenario])" for row in eachrow(data)
            @test row[:allocs] == 0
        end
    end
end;

@testset verbose = true "Overloaded inputs" begin
    backend = AutoForwardDiff()
    sparse_backend = MyAutoSparse(AutoForwardDiff())

    # Derivative
    x = 1.0
    y = [1.0, 1.0]
    @test DI.overloaded_input_type(prepare_derivative(copy, backend, x)) ==
        ForwardDiff.Dual{ForwardDiff.Tag{typeof(copy),Float64},Float64,1}
    @test DI.overloaded_input_type(prepare_derivative(copyto!, y, backend, x)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!),Float64},Float64,1}}

    # Gradient
    x = [1.0, 1.0]
    @test DI.overloaded_input_type(prepare_gradient(sum, backend, x)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(sum),Float64},Float64,2}}

    # Jacobian
    x = [1.0, 0.0, 0.0]
    @test DI.overloaded_input_type(prepare_jacobian(copy, backend, x)) ==
        ForwardDiff.Dual{ForwardDiff.Tag{typeof(copy),Float64},Float64,3}
    @test DI.overloaded_input_type(prepare_jacobian(copyto!, similar(x), backend, x)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!),Float64},Float64,3}}
    @test DI.overloaded_input_type(
        prepare_jacobian(copyto!, similar(x), sparse_backend, x)
    ) == Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!),Float64},Float64,1}}
end;
