using Pkg
Pkg.add(["ForwardDiff", "ReverseDiff"])  # ForwardDiff already in ReverseDiff's deps

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using StaticArrays: StaticArrays
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoReverseDiff(; compile=false), AutoReverseDiff(; compile=true)]
second_order_backends = [SecondOrder(AutoForwardDiff(), AutoReverseDiff())]

for backend in vcat(backends, second_order_backends)
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense

test_differentiation(
    vcat(backends, second_order_backends),
    default_scenarios(; include_constantified=true);
    logging=LOGGING,
);

test_differentiation(backends, static_scenarios(); logging=LOGGING);

## Sparse

test_differentiation(
    MyAutoSparse.(vcat(backends, second_order_backends)),
    sparse_scenarios();
    sparsity=true,
    logging=LOGGING,
);

@testset verbose = true "Overloaded inputs" begin
    backend = AutoReverseDiff()

    # Derivative
    x = 1.0
    @test_skip DI.overloaded_input(prepare_derivative(copy, backend, x)) isa
        ReverseDiff.TrackedArray{Float64,Float64,1,Vector{Float64},Vector{Float64}}

    # Gradient
    x = [1.0; 0.0; 0.0]
    @test DI.overloaded_input(prepare_gradient(sum, backend, x)) isa
        ReverseDiff.TrackedArray{Float64,Float64,1,Vector{Float64},Vector{Float64}}

    # Jacobian
    @test DI.overloaded_input(prepare_jacobian(copy, backend, x)) isa
        ReverseDiff.TrackedArray{Float64,Float64,1,Vector{Float64},Vector{Float64}}
    @test DI.overloaded_input(prepare_jacobian(copyto!, similar(x), backend, x)) isa
        ReverseDiff.TrackedArray{Float64,Float64,1,Vector{Float64},Vector{Float64}}
end;
