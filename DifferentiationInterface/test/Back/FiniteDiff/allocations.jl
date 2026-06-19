using StaticArrays: @SVector

@testset "allocations checks" begin
    function doit()
        backend = AutoFiniteDiff()
        x = @SVector [1.0, 2.0]
        tx = (2.0.*x,)
        f(x) = @. 3.0 * x
        prep = DifferentiationInterface.prepare_pushforward(f, backend, x, tx);
        return prep
    end
    doit()
    allocs = @allocated prep = doit()
    @test allocs == 0
end
