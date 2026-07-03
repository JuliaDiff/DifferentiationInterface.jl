using ADTypes
using ADTypes: dense_ad, mode
using DifferentiationInterface
using DifferentiationInterface:
    AutoSimpleFiniteDiff,
    AutoForwardFromPrimitive,
    AutoReverseFromPrimitive,
    inner,
    outer,
    forward_backend,
    reverse_backend,
    inplace_support,
    pushforward_performance,
    pullback_performance,
    hvp_mode,
    forward_counterpart,
    reverse_counterpart
import DifferentiationInterface as DI
using Test

fb = AutoSimpleFiniteDiff()
rb = AutoReverseFromPrimitive(AutoSimpleFiniteDiff())

@testset "NoAutoDiff" begin
    @test_throws NoAutoDiffSelectedError check_available(NoAutoDiff())
    @test_throws NoAutoDiffSelectedError inplace_support(NoAutoDiff())
    @test_throws NoAutoDiffSelectedError forward_counterpart(NoAutoDiff())
    @test_throws NoAutoDiffSelectedError reverse_counterpart(NoAutoDiff())
end

@testset "SecondOrder" begin
    backend = SecondOrder(fb, rb)
    @test check_available(backend)
    @test outer(backend) isa AutoSimpleFiniteDiff
    @test inner(backend) isa AutoReverseFromPrimitive
    @test mode(backend) isa ADTypes.ForwardMode
    @test Bool(inplace_support(backend))
    @test_throws ArgumentError pushforward_performance(backend)
    @test_throws ArgumentError pullback_performance(backend)
    @test_throws ArgumentError forward_counterpart(backend)
    @test_throws ArgumentError reverse_counterpart(backend)
end

@testset "MixedMode" begin
    backend = MixedMode(fb, rb)
    @test check_available(backend)
    @test mode(backend) isa DifferentiationInterface.ForwardAndReverseMode
    @test forward_backend(backend) isa AutoSimpleFiniteDiff
    @test reverse_backend(backend) isa AutoReverseFromPrimitive
    @test Bool(inplace_support(backend))
    @test_throws MethodError pushforward_performance(backend)
    @test_throws MethodError pullback_performance(backend)
    # a MixedMode backend is its own counterpart in both directions
    @test forward_counterpart(backend) === backend
    @test reverse_counterpart(backend) === backend
end

@testset "Counterparts" begin
    # forward-/reverse-mode backends are their own counterpart
    @test forward_counterpart(fb) === fb
    @test reverse_counterpart(rb) === rb
    # without a known counterpart, the backend is returned unchanged, with a warning
    @test (@test_logs (:warn, r"reverse-mode counterpart") reverse_counterpart(fb)) === fb
    # FromPrimitive wrappers swap the primitive, applying the counterpart inside
    @test forward_counterpart(rb) isa AutoForwardFromPrimitive{<:Any, <:AutoSimpleFiniteDiff}
    @test reverse_counterpart(forward_counterpart(rb)) isa
        AutoReverseFromPrimitive{<:Any, <:AutoSimpleFiniteDiff}
end

@testset "Sparse" begin
    for dense_backend in [fb, rb]
        backend = AutoSparse(dense_backend)
        @test mode(backend) == ADTypes.mode(dense_backend)
        @test Bool(inplace_support(backend))
        @test_throws ArgumentError pushforward_performance(backend)
        @test_throws ArgumentError pullback_performance(backend)
        @test_throws ArgumentError hvp_mode(backend)
    end
    # counterparts act on the dense backend and preserve the sparsity machinery
    backend = AutoSparse(rb)
    counterpart = forward_counterpart(backend)
    @test counterpart isa AutoSparse
    @test dense_ad(counterpart) isa AutoForwardFromPrimitive
    @test counterpart.sparsity_detector === backend.sparsity_detector
    @test counterpart.coloring_algorithm === backend.coloring_algorithm
end
