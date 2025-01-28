using ADTypes
using DifferentiationInterface
using DifferentiationInterface: required_packages
using Test

backend = SecondOrder(AutoForwardDiff(), AutoZygote())
@test string(backend) == "SecondOrder(AutoForwardDiff(), AutoZygote())"

detector = DenseSparsityDetector(AutoForwardDiff(); atol=1e-23)
@test string(detector) ==
    "DenseSparsityDetector(AutoForwardDiff(); atol=1.0e-23, method=:iterative)"

diffwith = DifferentiateWith(exp, AutoForwardDiff())
@test string(diffwith) == "DifferentiateWith(exp, AutoForwardDiff())"

@test required_packages(AutoForwardDiff()) == ["ForwardDiff"]
@test required_packages(AutoZygote()) == ["Zygote"]
@test required_packages(AutoSparse(AutoForwardDiff())) ==
    ["ForwardDiff", "SparseMatrixColorings"]
@test required_packages(SecondOrder(AutoForwardDiff(), AutoZygote())) ==
    ["ForwardDiff", "Zygote"]
@test required_packages(MixedMode(AutoForwardDiff(), AutoZygote())) ==
    ["ForwardDiff", "Zygote"]
