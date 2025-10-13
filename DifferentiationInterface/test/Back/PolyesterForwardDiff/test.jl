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

struct MyTag end

backends = [
    AutoPolyesterForwardDiff(; tag = ForwardDiff.Tag(MyTag(), Float64)),  #
    AutoPolyesterForwardDiff(; chunksize = 2),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
    @test DifferentiationInterface.inner_preparation_behavior(backend) isa
        DifferentiationInterface.PrepareInnerOverload
end

test_differentiation(
    backends,
    default_scenarios(;
        include_constantified = true, include_cachified = true, use_tuples = true
    );
    logging = LOGGING,
);

@testset "Batch size" begin
    @test DI.pick_batchsize(AutoPolyesterForwardDiff(), 10) ==
        DI.pick_batchsize(AutoForwardDiff(), 10)
    @test DI.pick_batchsize(AutoPolyesterForwardDiff(; chunksize = 3), rand(10)) ==
        DI.pick_batchsize(AutoForwardDiff(; chunksize = 3), rand(10))
end
