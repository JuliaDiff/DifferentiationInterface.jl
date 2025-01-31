using Pkg
Pkg.add(["ForwardDiff", "Zygote"])

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using JLArrays: JLArrays
using StaticArrays: StaticArrays
using Test
using Zygote: Zygote

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoZygote()]
second_order_backends = [SecondOrder(AutoForwardDiff(), AutoZygote())]

## Dense

@testset "Dense" begin
    test_differentiation(
        backends,
        default_scenarios(; include_constantified=true, include_cachified=true);
        excluded=[:second_derivative],
        logging=LOGGING,
    )

    test_differentiation(second_order_backends; logging=LOGGING)

    test_differentiation(
        backends[1],
        vcat(component_scenarios(), gpu_scenarios());
        excluded=SECOND_ORDER,
        logging=LOGGING,
    )
end

test_differentiation(
    AutoZygote(), complex_scenarios(); excluded=[:gradient, :jacobian], logging=LOGGING
);

## Sparse

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse.(vcat(backends, second_order_backends)),
        sparse_scenarios(; band_sizes=0:-1);
        sparsity=true,
        logging=LOGGING,
    )
end

## Errors

@testset "Errors" begin
    safe_log(x) = x > zero(x) ? log(x) : convert(typeof(x), NaN)
    @test_throws "Zygote failed to differentiate" derivative(safe_log, AutoZygote(), 0.0)
end
