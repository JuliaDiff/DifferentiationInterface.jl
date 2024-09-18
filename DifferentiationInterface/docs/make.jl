using Base: get_extension
using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter
using DocumenterMermaid
using DocumenterInterLinks

using ADTypes: ADTypes
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Zygote: Zygote

links = InterLinks(
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "SparseConnectivityTracer" => "https://adrianhill.de/SparseConnectivityTracer.jl/stable/",
    "SparseMatrixColorings" => "https://gdalle.github.io/SparseMatrixColorings.jl/stable/",
    "Symbolics" => "https://symbolics.juliasymbolics.org/stable/",
)

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[DifferentiationInterface],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(; assets=["assets/favicon.ico"]),
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["tutorials/basic.md", "tutorials/advanced.md"],
        "Explanation" => [
            "explanation/operators.md",
            "explanation/backends.md",
            "explanation/advanced.md",
        ],
        "api.md",
        "dev_guide.md",
    ],
    plugins=[links],
)

deploydocs(;
    repo="github.com/gdalle/DifferentiationInterface.jl",
    devbranch="main",
    dirname="DifferentiationInterface",
    tag_prefix="DifferentiationInterface-",
    push_preview=true,
)
