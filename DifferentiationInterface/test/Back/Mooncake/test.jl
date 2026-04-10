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

# Regression test for AutoMooncake gradient/gradient!/pullback/pushforward on
# a struct-backed AbstractArray (ComponentArray).  Before, the Mooncake
# extension returned the differential as a `Mooncake.Tangent` and DI tried to
# `copyto!` it into the preallocated `ComponentVector` buffer downstream
# callers (e.g. OptimizationBase) pass in, raising a `MethodError` on
# `iterate(::Mooncake.Tangent)`.  This blocked any Optimization.jl loop that
# used ComponentArrays parameters with `AutoMooncake`.
#
# The high-level scenario suite from DifferentiationInterfaceTest exercises
# the out-of-place and in-place versions of `gradient`, `pullback`, and
# `pushforward` for both `f(x)` and the `dy * f(x)` accumulation pattern,
# which together cover every code path the fix touches.
#
# `AutoMooncakeForward()` (without `friendly_tangents`) is excluded from this
# scenario because its forward-mode pushforward path has a separate,
# pre-existing bug at the *input* (Dual construction) side: it raises
# `ArgumentError: Tangent types do not match primal types` when given a
# `ComponentVector` `dx`, because Mooncake forward mode expects the tangent
# to already be a `Mooncake.Tangent` rather than a primal-shaped value.
# That input-side conversion is independent of the output-side fix in this
# PR; the friendly-tangents forward backend below covers the fixed code paths.
using ComponentArrays: ComponentArrays, ComponentVector
component_backends = [
    backends[1],  # AutoMooncake() — reverse, the path OptimizationBase uses
    backends[3],  # AutoMooncake(friendly_tangents=true) — reverse + friendly
    backends[4],  # AutoMooncakeForward(friendly_tangents=true) — forward + friendly
]
test_differentiation(
    component_backends,
    component_scenarios();
    excluded = SECOND_ORDER,
    logging = LOGGING,
)

# Direct gradient! sanity check on a small ComponentVector — this is the
# specific call shape OptimizationBase uses, kept as an explicit assertion in
# case `component_scenarios()` is ever pared down.
@testset "ComponentArrays gradient! into preallocated buffer" begin
    ps = ComponentVector(a = 1.0, b = [2.0, 3.0])
    myfun(p) = p.a^2 + sum(p.b .^ 2)
    for backend in component_backends
        gbuf = similar(ps)
        fill!(gbuf, 0)
        gradient!(myfun, gbuf, backend, ps)
        @test gbuf isa ComponentVector
        @test gbuf.a ≈ 2 * ps.a
        @test gbuf.b ≈ 2 .* ps.b
    end
end

test_differentiation(
    backends[3:4],
    nomatrix(static_scenarios());
    logging = LOGGING,
    excluded = SECOND_ORDER
)
