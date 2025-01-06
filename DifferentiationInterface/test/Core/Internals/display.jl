using ADTypes
using DifferentiationInterface
using Test

backend = SecondOrder(AutoForwardDiff(), AutoZygote())
@test string(backend) == "SecondOrder(AutoForwardDiff(), AutoZygote())"

detector = DenseSparsityDetector(AutoForwardDiff(); atol=1e-23)
@test string(detector) ==
    "DenseSparsityDetector(AutoForwardDiff(); atol=1.0e-23, method=:iterative)"

diffwith = DifferentiateWith(exp, AutoForwardDiff())
@test string(diffwith) == "DifferentiateWith(exp, AutoForwardDiff())"

@test DifferentiationInterface.package_name(AutoForwardDiff()) == "ForwardDiff"
@test DifferentiationInterface.package_name(AutoZygote()) == "Zygote"
@test DifferentiationInterface.package_name(AutoSparse(AutoForwardDiff())) == "ForwardDiff"
@test DifferentiationInterface.package_name(SecondOrder(AutoForwardDiff(), AutoZygote())) ==
    "ForwardDiff, Zygote"
