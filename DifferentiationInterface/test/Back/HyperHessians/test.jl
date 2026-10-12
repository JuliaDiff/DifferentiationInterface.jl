include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ExplicitImports
using HyperHessians
import JET
using Test

check_no_implicit_imports(DifferentiationInterface)

backends = [
    DI.AutoHyperHessians(),
    DI.AutoHyperHessians(; chunksize = 4),
    DI.AutoHyperHessians(; simd = true),
    DI.AutoHyperHessians(; jet = true),
    DI.AutoHyperHessians(; simd = true, jet = true),
]

for backend in backends
    @test DI.check_available(backend)
    @test !DI.check_inplace(backend)
    test_counterparts(backend)
end

@testset "Batch size" begin
    @test DI.pick_batchsize(DI.AutoHyperHessians(), rand(10)) isa DI.BatchSizeSettings
    @test DI.pick_batchsize(DI.AutoHyperHessians(), 10) isa DI.BatchSizeSettings
    @test DI.pick_batchsize(DI.AutoHyperHessians(; chunksize = 4), rand(10)) isa
        DI.BatchSizeSettings{4}
    @test DI.pick_batchsize(DI.AutoHyperHessians(; chunksize = 4), 10) isa
        DI.BatchSizeSettings{4}
end

scenarios = default_scenarios(; include_constantified = true, include_cachified = true)

test_differentiation(
    backends, scenarios;
    excluded = FIRST_ORDER, logging = LOGGING,
)

test_differentiation(
    DI.AutoHyperHessians(), scenarios;
    correctness = false,
    type_stability = safetypestab(:prepared),
    excluded = FIRST_ORDER,
    logging = LOGGING,
)
