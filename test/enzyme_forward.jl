using ADTypes: AutoEnzyme
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Enzyme: Enzyme

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoEnzyme(Enzyme.Forward))
@test supports_mutation(AutoEnzyme(Enzyme.Forward))

test_operators(
    AutoEnzyme(Enzyme.Forward); second_order=false, excluded=[:jacobian_allocating]
);
test_operators(AutoEnzyme(Enzyme.Forward), [:jacobian_allocating]; type_stability=false);
