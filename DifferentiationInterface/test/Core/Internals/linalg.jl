using DifferentiationInterface: recursive_similar, get_pattern
using SparseArrays
using Test

@testset "Recursive similar" begin
    @test recursive_similar(ones(Int, 2), Float32) isa Vector{Float32}
    @test recursive_similar((ones(Int, 2), ones(Bool, 3, 4)), Float32) isa
        Tuple{Vector{Float32},Matrix{Float32}}
    @test recursive_similar((a=ones(Int, 2), b=(ones(Bool, 3, 4),)), Float32) isa
        @NamedTuple{a::Vector{Float32}, b::Tuple{Matrix{Float32}}}
    @test_throws MethodError recursive_similar(1, Float32)
end

@testset "Sparsity pattern" begin
    D = Diagonal(rand(10))
    @test_broken get_pattern(D) == Diagonal(trues(10))
    @test get_pattern(sparse(D)) == Diagonal(trues(10))
end
