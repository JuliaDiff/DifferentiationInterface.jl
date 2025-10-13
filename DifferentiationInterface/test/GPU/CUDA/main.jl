@info "Testing on CUDA"
using Pkg
Pkg.add("CUDA")
Pkg.develop(PackageSpec(; path = "./DifferentiationInterface"))
using Test

@testset verbose = true "Simple" begin
    include("simple.jl")
end
