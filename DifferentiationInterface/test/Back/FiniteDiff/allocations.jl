using StaticArrays: @SVector

@testset "allocations checks" begin
    function pushforward_allocs()
        backend = AutoFiniteDiff()
        x = @SVector [1.0, 2.0]
        tx = (2.0 .* x,)
        f(x) = @. 3.0 * x
        prep = DifferentiationInterface.prepare_pushforward(f, backend, x, tx)
        return prep
    end
    pushforward_allocs()
    allocs = @allocated prep = pushforward_allocs()
    # This needs https://github.com/JuliaDiff/FiniteDiff.jl/pull/216 to be released.
    # Should be FiniteDiff v2.31.1.
    @test_broken allocs == 0

    function derivative_allocs()
        backend = AutoFiniteDiff()
        x = 3.0
        f(x) = x .* (@SVector [1.0, 2.0])
        prep = DifferentiationInterface.prepare_derivative(f, backend, x)
        return prep
    end
    derivative_allocs()
    allocs = @allocated prep = derivative_allocs()
    @test allocs == 0

    function jacobian_allocs()
        backend = AutoFiniteDiff()
        x = @SVector [1.0, 2.0]
        f(x) = 3.0 .* x
        prep = DifferentiationInterface.prepare_jacobian(f, backend, x)
        return prep
    end
    jacobian_allocs()
    allocs = @allocated prep = jacobian_allocs()
    # Using FiniteDiff.jl with StaticArrays to calculate a Jacobian does result in some allocations, apparently because the `FiniteDiff.JacobianCache` is a `mutable struct`.
    @test_broken allocs == 0
end
