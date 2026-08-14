using DifferentiationInterface
using Test

include("testutils.jl")

## Main tests

if haskey(ENV, "JULIA_DI_TEST_GROUP")
    folders = [ENV["JULIA_DI_TEST_GROUP"]]
else
    folders = ["Internals", "SimpleFiniteDiff", "ZeroBackends"]
end

@time @testset verbose = true "DifferentiationInterface.jl (Core)" begin
    @testset verbose = true "$folder" for folder in folders
        @testset verbose = true "$file" for file in readdir(
                joinpath(@__DIR__, "Core", folder)
            )
            endswith(file, ".jl") || continue
            @info "Testing $folder/$file"
            include(joinpath(@__DIR__, "Core", folder, file))
            yield()
        end
    end
end;
