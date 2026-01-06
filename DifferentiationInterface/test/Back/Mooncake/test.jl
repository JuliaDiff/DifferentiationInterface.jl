include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)


backends = [
    AutoMooncake(; config = nothing),
    AutoMooncake(; config = Mooncake.Config()),
    AutoMooncakeForward(; config = nothing),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

test_differentiation(
    backends,
    default_scenarios(;
        include_constantified = true, include_cachified = true, use_tuples = true
    );
    excluded = SECOND_ORDER,
    logging = LOGGING,
);

EXCLUDED = @static if VERSION ≥ v"1.11-" && VERSION ≤ v"1.12-"
    # testing only :hessian on 1.11 due to an opaque closure bug.  
    # This is potentially the same issue as discussed in 
    # https://github.com/chalk-lab/MistyClosures.jl/pull/12#issue-3278662295
    [FIRST_ORDER..., :hvp, :second_derivative]
else
    [FIRST_ORDER...]
end

# Test second-order differentiation (forward-over-reverse)
test_differentiation(
    [SecondOrder(AutoMooncakeForward(; config=nothing), AutoMooncake(; config=nothing))],
    excluded=EXCLUDED,
    logging=true,
)

@testset "NamedTuples" begin
    ps = (; A = rand(5), B = rand(5))
    myfun(ps) = sum(ps.A .* ps.B)
    grad = gradient(myfun, backends[1], ps)
    @test grad.A == ps.B
    @test grad.B == ps.A
end
