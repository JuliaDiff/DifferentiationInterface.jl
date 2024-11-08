using ADTypes
using ADTypes: mode
using DifferentiationInterface
using DifferentiationInterface:
    inner,
    outer,
    forward_backend,
    reverse_backend,
    inplace_support,
    pushforward_performance,
    pullback_performance,
    hvp_mode
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Test

@testset "SecondOrder" begin
    backend = SecondOrder(AutoForwardDiff(), AutoZygote())
    @test ADTypes.mode(backend) isa ADTypes.ForwardMode
    @test outer(backend) isa AutoForwardDiff
    @test inner(backend) isa AutoZygote
    @test mode(backend) isa ADTypes.ForwardMode
    @test !Bool(inplace_support(backend))
    @test_throws ArgumentError pushforward_performance(backend)
    @test_throws ArgumentError pullback_performance(backend)
    @test check_available(backend)
end

@testset "MixedMode" begin
    backend = MixedMode(AutoForwardDiff(), AutoZygote())
    @test ADTypes.mode(backend) isa DifferentiationInterface.ForwardAndReverseMode
    @test forward_backend(backend) isa AutoForwardDiff
    @test reverse_backend(backend) isa AutoZygote
    @test !Bool(inplace_support(backend))
    @test_throws MethodError pushforward_performance(backend)
    @test_throws MethodError pullback_performance(backend)
    @test check_available(backend)
end

@testset "Sparse" begin
    for dense_backend in [AutoForwardDiff(), AutoZygote()]
        backend = AutoSparse(dense_backend)
        @test ADTypes.mode(backend) == ADTypes.mode(dense_backend)
        @test check_available(backend) == check_available(dense_backend)
        @test inplace_support(backend) == inplace_support(dense_backend)
        @test_throws ArgumentError pushforward_performance(backend)
        @test_throws ArgumentError pullback_performance(backend)
        @test_throws ArgumentError hvp_mode(backend)
    end
end
