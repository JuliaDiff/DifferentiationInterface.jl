using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using LinearAlgebra
using ComponentArrays: ComponentArrays
using JLArrays: JLArrays
using SparseMatrixColorings
using StaticArrays: StaticArrays
using Test
using JET
import JET, Chairmarks

zero_backends = [AutoZeroForward(), AutoZeroReverse()]

for backend in zero_backends
    @test check_available(backend)
    @test check_inplace(backend)
end

@testset "Type stability" begin
    test_differentiation(
        [
            AutoZeroForward(),
            AutoZeroReverse(),
            SecondOrder(AutoZeroForward(), AutoZeroReverse()),
            SecondOrder(AutoZeroReverse(), AutoZeroForward()),
        ],
        default_scenarios(; include_batchified = false, include_constantified = true);
        correctness = false,
        type_stability = safetypestab(:full),
        logging = LOGGING,
    )

    test_differentiation(
        AutoSparse.(zero_backends, coloring_algorithm = GreedyColoringAlgorithm()),
        default_scenarios(; include_constantified = true);
        correctness = false,
        type_stability = safetypestab(:full),
        excluded = [
            :pushforward, :pullback, :gradient, :derivative, :hvp, :second_derivative,
        ],
        logging = LOGGING,
    )
end

@testset "Weird arrays" begin
    test_differentiation(
        [AutoZeroForward(), AutoZeroReverse()],
        zero.(vcat(component_scenarios(), static_scenarios(), gpu_scenarios()));
        correctness = true,
        logging = LOGGING,
    )
end

@testset "Empty arrays" begin
    test_differentiation(
        [AutoZeroForward(), AutoZeroReverse()], empty_scenarios(); excluded = [:jacobian]
    )
end;
