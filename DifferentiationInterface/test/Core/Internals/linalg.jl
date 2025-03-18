using DifferentiationInterface: recursive_similar
using Test

@test recursive_similar(ones(Int, 2), Float32) isa Vector{Float32}
@test recursive_similar((ones(Int, 2), ones(Bool, 3, 4)), Float32) isa
    Tuple{Vector{Float32},Matrix{Float32}}
@test recursive_similar((a=ones(Int, 2), b=(ones(Bool, 3, 4),)), Float32) isa
    @NamedTuple{a::Vector{Float32}, b::Tuple{Matrix{Float32}}}
@test_throws MethodError recursive_similar(1, Float32)
