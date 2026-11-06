using DifferentiationInterface
using DifferentiationInterface: AutoSimpleFiniteDiff
using Test

@testset "gradient_and_hvp is deprecated" begin
    f(x) = sum(abs2, x)
    backend = AutoSimpleFiniteDiff()
    x = [1.0, 2.0]
    tx = ([1.0, 0.0], [0.0, 1.0])
    prep = prepare_hvp(f, backend, x, tx)

    y, grad, tg = value_gradient_and_hvp(f, prep, backend, x, tx)
    @test y ≈ 5
    @test grad ≈ [2, 4]
    @test all(isapprox.(tg, ([2, 0], [0, 2]); rtol = 1.0e-3))

    grad_dep, tg_dep = @test_deprecated gradient_and_hvp(f, prep, backend, x, tx)
    @test grad_dep ≈ grad
    @test all(tg_dep .≈ tg)

    grad_in, tg_in = similar(x), map(similar, tx)
    grad_dep!, tg_dep! = @test_deprecated gradient_and_hvp!(
        f, grad_in, tg_in, prep, backend, x, tx
    )
    @test grad_dep! === grad_in
    @test tg_dep! === tg_in
    @test grad_dep! ≈ grad
    @test all(tg_dep! .≈ tg)
end
