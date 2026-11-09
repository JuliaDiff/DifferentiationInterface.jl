include("../../testutils.jl")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
using Piste: Piste
using Test
import JET

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

backends = [
    AutoPiste(),
    AutoPiste(; chunksize = 2),
    AutoPiste(; mode = ADTypes.ReverseMode()),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
    test_counterparts(backend)
end

@testset "Dense" begin
    test_differentiation(
        backends,
        default_scenarios(; include_constantified = true);
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )

    test_differentiation(
        backends,
        default_scenarios(;
            include_normal = false,
            include_batchified = false,
            include_cachified = true,
            include_constantorcachified = true,
            use_tuples = true,
            include_smaller = true,
        );
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )

    test_differentiation(
        backends;
        correctness = false,
        type_stability = safetypestab(:prepared),
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )
end

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse.(backends),
        sparse_scenarios(; include_constantified = true);
        sparsity = true,
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )
end
