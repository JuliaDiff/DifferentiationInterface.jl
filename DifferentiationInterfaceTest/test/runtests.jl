using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using Pkg
using SparseConnectivityTracer
using Test

GROUP = get(ENV, "JULIA_DIT_TEST_GROUP", "All")

## Main tests

@testset verbose = true "DifferentiationInterfaceTest.jl" begin
    if GROUP == "Formalities" || GROUP == "All"
        @testset verbose = true "Formalities" begin
            include("formalities.jl")
        end
        @testset verbose = true "Scenarios" begin
            include("scenario.jl")
        end
    end

    if GROUP == "Zero" || GROUP == "All"
        @testset verbose = true "Zero" begin
            include("zero_backends.jl")
        end
    end

    if GROUP == "Standard" || GROUP == "All"
        @testset verbose = true "Standard" begin
            include("standard.jl")
        end
    end

    if GROUP == "Weird" || GROUP == "All"
        @testset verbose = true "Weird" begin
            include("weird.jl")
        end
    end
end;
