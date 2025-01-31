using DifferentiationInterface
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using Test

scen = Scenario{:gradient,:out}(
    sum, zeros(10); res1=ones(10), name="My pretty little scenario"
)
@test string(scen) == "My pretty little scenario"

testset = test_differentiation(
    AutoForwardDiff(), [scen]; testset_name="My amazing test set"
)

data = benchmark_differentiation(
    AutoForwardDiff(), [scen]; testset_name="My amazing test set"
)
