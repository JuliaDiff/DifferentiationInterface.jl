using DifferentiationInterface
using Test

include("testutils.jl")

## Main tests

if haskey(ENV, "JULIA_DI_TEST_GROUP")
    folders = [
        ENV["JULIA_DI_TEST_GROUP"],
    ]
else
    folders = [
        joinpath("Core", "Internals"),
        joinpath("Core", "SimpleFiniteDiff"),
        joinpath("Core", "ZeroBackends"),
    ]
end

@time @testset verbose = true "DifferentiationInterface.jl" begin
    @testset verbose = true "$folder" for folder in folders
        @testset verbose = true "$file" for file in readdir(joinpath(@__DIR__, folder))
            endswith(file, ".jl") || continue
            @info "Testing $folder/$file"
            include(joinpath(@__DIR__, folder, file))
            yield()
        end
    end
end;
