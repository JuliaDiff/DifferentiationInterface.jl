using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Symbolics: Symbolics
using Zygote: Zygote

coloring_algorithm = DI.GreedyColoringAlgorithm()
sparsity_detector = DI.SymbolicsSparsityDetector()

backends = [
    AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoZygote()); sparsity_detector, coloring_algorithm
    ),
]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(
    backends, sparse_scenarios(); first_order=false, sparsity=true, logging=LOGGING
);
