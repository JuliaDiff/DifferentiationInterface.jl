using CUDA
using DifferentiationInterface
import DifferentiationInterface as DI
using LinearAlgebra
using Test

CUDA.versioninfo()

@testset "Basis" begin
    x = CuVector(rand(Float32, 3))
    b = DI.basis(x, 2)
    @test Array(b) == [0, 1, 0]

    X = CuMatrix(rand(Float32, 2, 2))
    B = DI.multibasis(X, [2, 3])
    @test Array(B) == [0 1; 1 0]
end

@testset "Jacobian" begin
    x = CuVector(rand(Float32, 3))
    backend = DI.AutoSimpleFiniteDiff()
    J = jacobian(identity, backend, x)
    @test J == I
end
