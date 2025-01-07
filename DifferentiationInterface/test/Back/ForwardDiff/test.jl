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

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

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
        @test DI.pick_batchsize(AutoForwardDiff(; chunksize=5), rand(7)) isa
            DI.BatchSizeSettings{5}
        @test (@inferred DI.pick_batchsize(AutoForwardDiff(), @SVector(rand(7)))) isa
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

## Overloaded inputs
@testset verbose = true "Overload inputs" begin
    backend = AutoForwardDiff()
    u0 = [1.0; 0.0; 0.0]
    # One Arg
    prep = DI.prepare_jacobian(nothing, backend, u0)
    @test typeof(overloaded_inputs(prep)) == ForwardDiff.Dual{Nothing,Float64,3}
    # Two Arg
    prep = DI.prepare_jacobian(nothing, similar(u0), backend, u0)
    @test typeof(overloaded_inputs(prep)) == Vector{ForwardDiff.Dual{Nothing,Float64,3}}

    # Sparse backend
    backend = MyAutoSparse(AutoForwardDiff())
    prep = DI.prepare_jacobian(copyto!, similar(u0), backend, u0)
    @test typeof(overloaded_inputs(prep)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!),Float64},Float64,1}}
end;
