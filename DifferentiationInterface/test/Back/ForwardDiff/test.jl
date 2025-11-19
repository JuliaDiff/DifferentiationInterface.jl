using Pkg
Pkg.activate(@__DIR__)
include("../../testutils.jl")

using ADTypes: ADTypes
using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using StaticArrays: StaticArrays, @SVector
using JLArrays: JLArrays
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

struct MyTag end

backends = [
    AutoForwardDiff(),
    AutoForwardDiff(; chunksize = 5),
    AutoForwardDiff(; tag = ForwardDiff.Tag(MyTag(), Float64)),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

@testset "Dense" begin
    test_differentiation(
        backends, default_scenarios(; include_constantified = true); logging = LOGGING
    )

    test_differentiation(
        AutoForwardDiff(),
        default_scenarios(;
            include_normal = false,
            include_batchified = false,
            include_cachified = true,
            include_constantorcachified = true,
            use_tuples = true,
            include_smaller = true,
        );
        logging = LOGGING,
    )

    test_differentiation(
        AutoForwardDiff();
        correctness = false,
        type_stability = safetypestab(:prepared),
        logging = LOGGING,
    )

    test_differentiation(
        AutoForwardDiff(; chunksize = 5);
        correctness = false,
        type_stability = safetypestab(:full),
        excluded = [:hessian],
        logging = LOGGING,
    )
end

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse(AutoForwardDiff()), default_scenarios(); logging = LOGGING
    )

    test_differentiation(
        MyAutoSparse(AutoForwardDiff()),
        sparse_scenarios(; include_constantified = true);
        sparsity = true,
        logging = LOGGING,
    )
end

@testset "Weird" begin
    test_differentiation(AutoForwardDiff(), component_scenarios(); logging = LOGGING)
    test_differentiation(AutoForwardDiff(), static_scenarios(); logging = LOGGING)
    test_differentiation(
        DI.AutoForwardFromPrimitive(AutoForwardDiff()), gpu_scenarios(); logging = LOGGING
    )

    @testset "Batch size" begin
        @test DI.pick_batchsize(AutoForwardDiff(), rand(7)) isa DI.BatchSizeSettings{7}
        @test DI.pick_batchsize(AutoForwardDiff(; chunksize = 5), rand(7)) isa
            DI.BatchSizeSettings{5}
        @test (@inferred DI.pick_batchsize(AutoForwardDiff(), @SVector(rand(7)))) isa
            DI.BatchSizeSettings{7}
        @test (
            @inferred DI.pick_batchsize(
                AutoForwardDiff(; chunksize = 5), @SVector(rand(7))
            )
        ) isa DI.BatchSizeSettings{5}
    end
end

@testset verbose = true "Overloaded inputs" begin
    backend = AutoForwardDiff()
    sparse_backend = MyAutoSparse(AutoForwardDiff())

    # Derivative
    x = 1.0
    y = [1.0, 1.0]
    @test DI.overloaded_input_type(prepare_derivative(copy, backend, x)) ==
        ForwardDiff.Dual{ForwardDiff.Tag{typeof(copy), Float64}, Float64, 1}
    @test DI.overloaded_input_type(prepare_derivative(copyto!, y, backend, x)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!), Float64}, Float64, 1}}

    # Gradient
    x = [1.0, 1.0]
    @test DI.overloaded_input_type(prepare_gradient(sum, backend, x)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(sum), Float64}, Float64, 2}}

    # Jacobian
    x = [1.0, 0.0, 0.0]
    @test DI.overloaded_input_type(prepare_jacobian(copy, backend, x)) ==
        ForwardDiff.Dual{ForwardDiff.Tag{typeof(copy), Float64}, Float64, 3}
    @test DI.overloaded_input_type(prepare_jacobian(copyto!, similar(x), backend, x)) ==
        Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!), Float64}, Float64, 3}}
    @test DI.overloaded_input_type(
        prepare_jacobian(copyto!, similar(x), sparse_backend, x)
    ) == Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(copyto!), Float64}, Float64, 1}}
end;

include("benchmark.jl")
