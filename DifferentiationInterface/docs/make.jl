using Base: get_extension
using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter
using DocumenterMermaid
using DocumenterInterLinks

using ADTypes: ADTypes
using ForwardDiff: ForwardDiff
using Zygote: Zygote

links = InterLinks(
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "SparseConnectivityTracer" => "https://adrianhill.de/SparseConnectivityTracer.jl/stable/",
    "SparseMatrixColorings" => "https://gdalle.github.io/SparseMatrixColorings.jl/stable/",
    "Symbolics" => "https://symbolics.juliasymbolics.org/stable/",
)

readme_str = read(joinpath(@__DIR__, "..", "README.md"), String)
readme_str = replace(readme_str, "> [!CAUTION]\n> " => "!!! warning\n    ")
write(joinpath(@__DIR__, "src", "index.md"), readme_str)

makedocs(;
    modules = [DifferentiationInterface],
    authors = "Guillaume Dalle, Adrian Hill",
    sitename = "DifferentiationInterface.jl",
    format = Documenter.HTML(; assets = ["assets/favicon.ico"]),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorials/basic.md", "tutorials/advanced.md"],
        "Explanation" => [
            "explanation/operators.md",
            "explanation/backends.md",
            "explanation/advanced.md",
        ],
        "FAQ" => ["faq/limitations.md", "faq/differentiability.md"],
        "api.md",
        "dev_guide.md",
    ],
    plugins = [links],
)

deploydocs(;
    repo = "github.com/JuliaDiff/DifferentiationInterface.jl",
    devbranch = "main",
    dirname = "DifferentiationInterface",
    tag_prefix = "DifferentiationInterface-",
    push_preview = false,
)
