using ADTypes
using ADTypes: mode
using DifferentiationInterface
using DifferentiationInterface:
    AutoSimpleFiniteDiff,
    AutoReverseFromPrimitive,
    inner,
    outer,
    forward_backend,
    reverse_backend,
    inplace_support,
    pushforward_performance,
    pullback_performance,
    hvp_mode
import DifferentiationInterface as DI
using Test

fb = AutoSimpleFiniteDiff()
rb = AutoReverseFromPrimitive(AutoSimpleFiniteDiff())

@testset "NoAutoDiff" begin
    @test_throws NoAutoDiffSelectedError check_available(NoAutoDiff())
    @test_throws NoAutoDiffSelectedError inplace_support(NoAutoDiff())
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
end
