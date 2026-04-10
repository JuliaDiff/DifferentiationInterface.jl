include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using LinearAlgebra: Hermitian, SymTridiagonal, Symmetric
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

# friendly_tangents + StaticArrays broken on Julia 1.11 (upstream Mooncake bug)
@static if !(VERSION ≥ v"1.11-" && VERSION < v"1.12-")
    test_differentiation(
        backends[3:4],
        nomatrix(static_scenarios());
        logging = LOGGING,
        excluded = SECOND_ORDER,
    )
end

@testset "Friendly tangents structured matrices" begin
    backend = AutoMooncake(; config = Mooncake.Config(; friendly_tangents = true))
    inputs = (
        Symmetric([2.0 1.0; 1.0 3.0]),
        Hermitian(ComplexF64[2 1 + im; 1 - im 3]),
        SymTridiagonal([2.0, 3.0, 4.0], [5.0, 6.0]),
    )
    f(x) = real(sum(abs2, x))

    @testset "$(typeof(x))" for x in inputs
        grad = gradient(f, backend, x)
        y, grad2 = value_and_gradient(f, backend, x)
        pb = only(pullback(identity, backend, x, (x,)))

        @test grad isa Matrix
        @test grad2 isa Matrix
        @test pb isa Matrix
        @test grad == grad2
        @test y == f(x)
        @test pb == Matrix(x)

        grad_dense = zero(Matrix(x))
        @test gradient!(f, grad_dense, backend, x) === grad_dense
        @test grad_dense == grad

        tx_dense = (zero(Matrix(x)),)
        @test only(pullback!(identity, tx_dense, backend, x, (x,))) === tx_dense[1]
        @test tx_dense[1] == pb
    end
end
