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

test_differentiation(
    backends[3:4],
    nomatrix(static_scenarios());
    logging = LOGGING,
    excluded = SECOND_ORDER,
)

@testset "Friendly tangents structured matrices" begin
    # Mooncake 0.5.25+ returns a plain `Matrix` for structured inputs under
    # `friendly_tangents=true` (chalk-lab/Mooncake.jl#1103); the complex case
    # follows the standard reverse-mode convention (chalk-lab/Mooncake.jl#773).
    #
    # Per-wrapper test functions are chosen for their non-triviality given
    # that matmul on small matrices hits a `utf8proc_isupper` ccall that
    # Mooncake cannot differentiate (LinearAlgebra._matmul2x2_elements →
    # WrapperChar). For the real wrappers we use a manual triple-loop tr(X^3)
    # whose unrestricted gradient is 3·X²; for Hermitian we use the simpler
    # abs2-sum because the complex Wirtinger ground truth via tr(X^3) is
    # convention-heavy. The expected friendly gradient is then computed by
    # aggregating the unrestricted per-element gradient into the wrapper's
    # canonical storage cells, derived independently of Mooncake.

    # tr(X^3) without matmul. Indices i,j,k each range over axes(X,1).
    function tr_x3(X)
        s = zero(eltype(X))
        n = size(X, 1)
        @inbounds for i in 1:n, j in 1:n, k in 1:n
            s += X[i, j] * X[j, k] * X[k, i]
        end
        return real(s)
    end

    # Symmetric storage: upper triangle holds the sum of (i,j) and (j,i)
    # per-element contributions; strict lower triangle is zero.
    function aggregate_symmetric(G)
        n = size(G, 1)
        H = zero(G)
        @inbounds for i in 1:n
            H[i, i] = G[i, i]
            for j in (i + 1):n
                H[i, j] = G[i, j] + G[j, i]
            end
        end
        return H
    end

    # SymTridiagonal storage: diagonal + symmetric off-diagonals (both
    # `(i,i+1)` and `(i+1,i)` slots hold the doubled contribution); entries
    # outside that band are structurally zero in the wrapper.
    function aggregate_symtridiagonal(G)
        n = size(G, 1)
        H = zero(G)
        @inbounds for i in 1:n
            H[i, i] = G[i, i]
            if i < n
                aggregated = G[i, i + 1] + G[i + 1, i]
                H[i, i + 1] = aggregated
                H[i + 1, i] = aggregated
            end
        end
        return H
    end

    backend = AutoMooncake(; config = Mooncake.Config(; friendly_tangents = true))
    abs2_sum(x) = real(sum(abs2, x))
    cases = (
        (
            x = Symmetric([2.0 1.0; 1.0 3.0]),
            f = tr_x3,
            expected_grad = let M = Matrix(Symmetric([2.0 1.0; 1.0 3.0]))
                aggregate_symmetric(3 * M^2)
            end,
        ),
        (
            x = Hermitian(ComplexF64[2 1 + im; 1 - im 3]),
            f = abs2_sum,
            expected_grad = ComplexF64[4 4 + 4im; 0 6],
        ),
        (
            x = SymTridiagonal([2.0, 3.0, 4.0], [5.0, 6.0]),
            f = tr_x3,
            expected_grad = let M = Matrix(SymTridiagonal([2.0, 3.0, 4.0], [5.0, 6.0]))
                aggregate_symtridiagonal(3 * M^2)
            end,
        ),
    )

    @testset "$(typeof(case.x))" for case in cases
        x = case.x
        f = case.f
        grad = gradient(f, backend, x)
        y, grad2 = value_and_gradient(f, backend, x)
        pb = only(pullback(identity, backend, x, (x,)))

        @test grad isa Matrix
        @test grad2 isa Matrix
        @test pb isa Matrix
        @test grad == grad2
        @test grad ≈ case.expected_grad
        @test y ≈ f(x)
        @test pb == Matrix(x)

        grad_dense = zero(Matrix(x))
        @test gradient!(f, grad_dense, backend, x) === grad_dense
        @test grad_dense == grad

        tx_dense = (zero(Matrix(x)),)
        @test only(pullback!(identity, tx_dense, backend, x, (x,))) === tx_dense[1]
        @test tx_dense[1] == pb
    end
end
