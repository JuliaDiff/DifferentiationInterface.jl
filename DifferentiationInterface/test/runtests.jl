using DifferentiationInterface
using Pkg
using Test

include("testutils.jl")

## Main tests

@time @testset verbose = true "DifferentiationInterface.jl" begin
    if haskey(ENV, "JULIA_DI_TEST_GROUP")
        category, folder = split(ENV["JULIA_DI_TEST_GROUP"], '/')
        @testset verbose = true "$category" begin
            @testset verbose = true "$folder" begin
                @testset verbose = true "$file" for file in readdir(
                        joinpath(@__DIR__, category, folder)
                    )
                    endswith(file, ".jl") || continue
                    @info "Testing $category/$folder/$file"
                    include(joinpath(@__DIR__, category, folder, file))
                    yield()
                end
            end
        end
    else
        category = "Core"
        @testset verbose = true for folder in readdir(joinpath(@__DIR__, category))
            isdir(joinpath(@__DIR__, category, folder)) || continue
            @testset verbose = true "$file" for file in readdir(
                    joinpath(@__DIR__, category, folder)
                )
                endswith(file, ".jl") || continue
                @info "Testing $category/$folder/$file"
                include(joinpath(@__DIR__, category, folder, file))
                yield()
            end
        end
    end
end;
