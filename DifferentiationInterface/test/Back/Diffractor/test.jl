include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using Diffractor: Diffractor
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

for backend in [AutoDiffractor()]
    @test check_available(backend)
    @test !check_inplace(backend)
end

@testset "Broadcast" begin
    # https://github.com/JuliaDiff/Diffractor.jl/issues/290
    x, dx = [1.0, 2.0], [3.0, 4.0]
    @test pushforward(x -> sin.(x), AutoDiffractor(), x, (dx,))[1] ≈ cos.(x) .* dx
end

test_differentiation(
    AutoDiffractor(),
    default_scenarios(; linalg = false);
    excluded = SECOND_ORDER,
    logging = LOGGING,
);
