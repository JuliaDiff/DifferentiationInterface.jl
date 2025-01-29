using Pkg
Pkg.add(["ForwardDiff", "PolyesterForwardDiff"])

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

backends = [
    AutoPolyesterForwardDiff(; tag=:hello),  #
    AutoPolyesterForwardDiff(; chunksize=2),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

@testset "Batch size" begin
    @test DI.pick_batchsize(AutoPolyesterForwardDiff(), 10) ==
        DI.pick_batchsize(AutoForwardDiff(), 10)
    @test DI.pick_batchsize(AutoPolyesterForwardDiff(; chunksize=3), rand(10)) ==
        DI.pick_batchsize(AutoForwardDiff(; chunksize=3), rand(10))
end
