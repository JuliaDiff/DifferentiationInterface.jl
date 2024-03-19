using DifferentiationInterface
using DifferentiationInterface: AutoFastDifferentiation
using DifferentiationInterface.DifferentiationTest
using FastDifferentiation: FastDifferentiation

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoFastDifferentiation())

test_operators(
    AutoFastDifferentiation();
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    second_order=false,
    type_stability=false,
);
