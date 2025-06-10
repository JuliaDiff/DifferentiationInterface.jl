@info "Testing on CUDA"
using Pkg
Pkg.add(["CUDA", "DifferentiationInterface"])
using Test

@testset verbose = true "Simple" begin
    include("simple.jl")
end
