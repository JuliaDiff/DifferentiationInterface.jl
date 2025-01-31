using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using ComponentArrays: ComponentArrays
using JLArrays: JLArrays
using SparseMatrixColorings
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

zero_backends = [AutoZeroForward(), AutoZeroReverse()]

@testset "Correctness" begin
    test_differentiation(zero_backends, map(zero, default_scenarios()); logging=LOGGING)
end

@testset "Type stability" begin
    test_differentiation(
        AutoZeroForward(),
        default_scenarios(; include_batchified=false, include_constantified=true);
        correctness=false,
        type_stability=:full,
        logging=LOGGING,
    )

    test_differentiation(
        AutoZeroReverse(),
        default_scenarios(; include_batchified=false, include_constantified=true);
        correctness=false,
        type_stability=:full,
        logging=LOGGING,
    )

    test_differentiation(
        [
            SecondOrder(AutoZeroForward(), AutoZeroReverse()),
            SecondOrder(AutoZeroReverse(), AutoZeroForward()),
        ],
        default_scenarios(; include_batchified=false, include_constantified=true);
        correctness=false,
        type_stability=:full,
        logging=LOGGING,
    )

    test_differentiation(
        AutoSparse.(zero_backends, coloring_algorithm=GreedyColoringAlgorithm()),
        default_scenarios(; include_constantified=true);
        correctness=false,
        type_stability=:full,
        excluded=[
            :pushforward, :pullback, :gradient, :derivative, :hvp, :second_derivative
        ],
        logging=LOGGING,
    )
end

@testset "Weird arrays" begin
    test_differentiation(
        [AutoZeroForward(), AutoZeroReverse()],
        zero.(vcat(component_scenarios(), static_scenarios(), gpu_scenarios()));
        correctness=true,
        logging=LOGGING,
    )
end
