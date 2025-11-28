using Test
group = ENV["JULIA_DI_TEST_GROUP"]
@testset "$group" begin
    include(joinpath(@__DIR__, group, "test.jl"))
end
