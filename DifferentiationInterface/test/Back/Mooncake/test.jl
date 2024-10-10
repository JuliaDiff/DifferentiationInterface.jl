using Pkg
Pkg.add("Mooncake")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    backends,
    default_scenarios(; include_constantified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
