using Pkg
Pkg.add("Mooncake")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

LOGGING = get(ENV, "CI", "false") == "false"

backends = [
    AutoMooncake(; config=nothing),
    AutoMooncake(; config=Mooncake.Config()),
    AutoMooncakeForward(; config=nothing),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    backends,
    default_scenarios(;
        include_constantified=true, include_cachified=true, use_tuples=true
    );
    excluded=SECOND_ORDER,
    logging=LOGGING,
);

@testset "NamedTuples" begin
    ps = (; A=rand(5), B=rand(5))
    myfun(ps) = sum(ps.A .* ps.B)
    grad = gradient(myfun, backends[1], ps)
    @test grad.A == ps.B
    @test grad.B == ps.A
end
