include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

nomatrix(scens) = filter(s -> !(s.x isa AbstractMatrix) && !(s.y isa AbstractMatrix), scens)

backends = [
    AutoMooncake(),
    AutoMooncakeForward(),
    AutoMooncake(; config = Mooncake.Config(; friendly_tangents = true)),
    AutoMooncakeForward(; config = Mooncake.Config(; friendly_tangents = true)),
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
    test_counterparts(backend)
end

@testset "Counterpart config" begin
    config = Mooncake.Config(; friendly_tangents = true)
    @test DifferentiationInterface.forward_counterpart(AutoMooncake()) === AutoMooncakeForward()
    @test DifferentiationInterface.reverse_counterpart(AutoMooncakeForward()) === AutoMooncake()
    # the Mooncake config must carry over to the counterpart
    @test DifferentiationInterface.forward_counterpart(AutoMooncake(; config)) ===
        AutoMooncakeForward(; config)
    @test DifferentiationInterface.reverse_counterpart(AutoMooncakeForward(; config)) ===
        AutoMooncake(; config)
end

test_differentiation(
    backends[3:4],
    default_scenarios();
    excluded = SECOND_ORDER,
    logging = LOGGING,
);

test_differentiation(
    backends[3:4],
    nomatrix(
        default_scenarios(;
            include_normal = false,
            include_constantified = true,
            include_cachified = true,
            use_tuples = true
        )
    );
    excluded = SECOND_ORDER,
    logging = LOGGING,
);

test_differentiation(
    backends[1:2],
    nomatrix(default_scenarios());
    excluded = SECOND_ORDER,
    logging = LOGGING,
);

EXCLUDED = @static if VERSION ≥ v"1.11-" && VERSION ≤ v"1.12-"
    # testing only :hessian on 1.11 due to an opaque closure bug.
    # this is potentially the same issue as discussed in
    # https://github.com/chalk-lab/MistyClosures.jl/pull/12#issue-3278662295
    [FIRST_ORDER..., :hvp, :second_derivative]
else
    [FIRST_ORDER...]
end

# Test second-order differentiation (forward-over-reverse)
test_differentiation(
    [SecondOrder(AutoMooncakeForward(), AutoMooncake())],
    nomatrix(default_scenarios());
    excluded = EXCLUDED,
    logging = LOGGING,
)

@testset "NamedTuples" begin
    ps = (; A = rand(5), B = rand(5))
    myfun(ps) = sum(ps.A .* ps.B)
    grad = gradient(myfun, backends[1], ps)
    @test grad.A == ps.B
    @test grad.B == ps.A
end

# see https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/986
if pkgversion(Mooncake) < v"0.5.25"
    test_differentiation(
        backends[3:4],
        nomatrix(static_scenarios());
        logging = LOGGING,
        excluded = SECOND_ORDER
    )
end
