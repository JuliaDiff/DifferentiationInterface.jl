using DifferentiationInterface: OneElement, basis
using LinearAlgebra
using StaticArrays, JLArrays
using Test

@testset "OneElement" begin
    a = rand(Float32, 3)
    i = eachindex(a)[2]
    oe = OneElement(i, 2.0f0, a)
    @test eltype(oe) == Float32
    @test oe == [0, 2, 0]

    a = rand(Float64, 3, 3)
    i = eachindex(a)[4]
    oe = OneElement(i, 2.0, a)
    @test eltype(oe) == Float64
    @test oe == [0 2 0; 0 0 0; 0 0 0]

    a = Diagonal(ones(3))
    i = eachindex(a)[4]
    oe = OneElement(i, 2.0, a)
    @test oe == [0 2 0; 0 0 0; 0 0 0]
end

@testset "Basis" begin
    b_ref = [0, 1, 0]
    @test basis(rand(3), 2) isa Vector
    @test basis(rand(3), 2) == b_ref
    @test basis(jl(rand(3)), 2) isa JLArray
    @test all(basis(jl(rand(3)), 2) .== b_ref)
    @test basis(@SVector(rand(3)), 2) isa SVector
    @test basis(@SVector(rand(3)), 2) == b_ref

    b_ref = [0 1 0; 0 0 0; 0 0 0]
    @test basis(rand(3, 3), 4) isa Matrix
    @test basis(rand(3, 3), 4) == b_ref
    @test basis(jl(rand(3, 3)), 4) isa JLArray
    @test all(basis(jl(rand(3, 3)), 4) .== b_ref)
    @test basis(@SMatrix(rand(3, 3)), 4) isa SMatrix
    @test basis(@SMatrix(rand(3, 3)), 4) == b_ref
end
