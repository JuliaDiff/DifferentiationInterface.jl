using DifferentiationInterface
using DifferentiationInterfaceTest
using Documenter
using DocumenterInterLinks

using BenchmarkTools: BenchmarkTools
using DataFrames: DataFrames
using ForwardDiff: ForwardDiff

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force = true)

makedocs(;
    modules = [DifferentiationInterfaceTest],
    authors = "Guillaume Dalle, Adrian Hill",
    sitename = "DifferentiationInterfaceTest.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md", #
        "Tutorial" => "tutorial.md", #
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaDiff/DifferentiationInterface.jl",
    devbranch = "main",
    dirname = "DifferentiationInterfaceTest",
    tag_prefix = "DifferentiationInterfaceTest-",
    push_preview = false,
)
