using DifferentiationInterface: basis, multibasis
using LinearAlgebra
using StaticArrays, JLArrays
using Test
using Dates

@testset "Basis" begin
    b_ref = [0, 1, 0]
    @test basis(rand(3), 2) isa Vector
    @test basis(rand(3), 2) == b_ref
    @test basis(jl(rand(3)), 2) isa JLArray
    @test Array(basis(jl(rand(3)), 2)) == [0, 1, 0]
    @test multibasis(jl(rand(3)), [1, 2]) isa JLArray
    @test Array(multibasis(jl(rand(3)), [1, 2])) == [1, 1, 0]
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

    t = [Time(1) - Time(0)]
    @test basis(t, 1) isa Vector{Nanosecond}

    @test basis([1, 2]) == [0, 0]
    @test basis(Int[]) == Int[]
end
