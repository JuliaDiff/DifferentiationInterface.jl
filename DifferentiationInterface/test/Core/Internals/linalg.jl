using DifferentiationInterface: recursive_similar, get_pattern, arroftup_to_tupofarr
using SparseArrays
using Test
using JLArrays, ComponentArrays

@testset "Recursive similar" begin
    @test recursive_similar(ones(Int, 2), Float32) isa Vector{Float32}
    @test recursive_similar((ones(Int, 2), ones(Bool, 3, 4)), Float32) isa
        Tuple{Vector{Float32}, Matrix{Float32}}
    @test recursive_similar((a = ones(Int, 2), b = (ones(Bool, 3, 4),)), Float32) isa
        @NamedTuple{a::Vector{Float32}, b::Tuple{Matrix{Float32}}}
    @test_throws MethodError recursive_similar(1, Float32)
end

@testset "Sparsity pattern" begin
    D = Diagonal(rand(10))
    @test_broken get_pattern(D) == Diagonal(trues(10))
    @test get_pattern(sparse(D)) == Diagonal(trues(10))
end

@testset "Wrong-mode array conversion" begin
    x = [1.0, 3.0, 5.0]
    xt = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    y = ComponentVector(a = [1.0, 3.0], b = [5.0])
    yt = ComponentVector(a = [(1.0, 2.0), (3.0, 4.0)], b = [(5.0, 6.0)])
    z = jl([1.0, 3.0, 5.0])
    zt = jl([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
    @test arroftup_to_tupofarr((1.0, 2.0), 1.0) == (1.0, 2.0)
    @test arroftup_to_tupofarr(xt, x) == ([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])
    @test arroftup_to_tupofarr(yt, y) == (ComponentVector(a = [1.0, 3.0], b = [5.0]), ComponentVector(a = [2.0, 4.0], b = [6.0]))
    @test arroftup_to_tupofarr(zt, z) == (jl([1.0, 3.0, 5.0]), jl([2.0, 4.0, 6.0]))
    @test arroftup_to_tupofarr(xt, x)[1] isa Vector
    @test arroftup_to_tupofarr(yt, y)[1] isa ComponentVector
    @test arroftup_to_tupofarr(zt, z)[1] isa JLVector
end
