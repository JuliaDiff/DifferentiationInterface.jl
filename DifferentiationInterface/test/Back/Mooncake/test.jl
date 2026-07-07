include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using Mooncake: Mooncake
using LinearAlgebra
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
end

test_differentiation(
    backends[3:4],
    default_scenarios();
    excluded = SECOND_ORDER,
    logging = LOGGING,
    testset_name = "Basics"
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
    testset_name = "Constantified and cachified"
);

test_differentiation(
    backends[1:2],
    nomatrix(default_scenarios());
    excluded = SECOND_ORDER,
    logging = LOGGING,
    testset_name = "No friendly tangents"
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
    testset_name = "Second order"
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
        excluded = SECOND_ORDER,
        testset_name = "Static scenarios"
    )
end

@testset "Closure over differentiable data" begin
    # https://github.com/chalk-lab/Mooncake.jl/issues/1238
    function make_f()
        data = [1.0, 2.0, 3.0]
        return p -> sum(abs2, LowerTriangular([p[1] 0 0; p[2] p[1] 0; p[3] p[2] p[1]] + I) \ data)
    end

    f = make_f()
    xA = [0.3, 0.5, 0.2]
    xB = [1.5, 2.0, -1.0]
    b = DI.AutoMooncake()
    g = DI.gradient(f, b, xB)
    prep = DI.prepare_gradient(f, b, xA)
    @test DI.gradient(f, prep, b, xB) ≈ g
    @test DI.gradient(f, prep, b, xB) ≈ g
end
