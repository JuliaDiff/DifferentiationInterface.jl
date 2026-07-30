using Base: get_extension
using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter
using DocumenterInterLinks

using ADTypes: ADTypes
using ForwardDiff: ForwardDiff
using Zygote: Zygote

links = InterLinks(
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "SparseConnectivityTracer" => "https://adrianhill.de/SparseConnectivityTracer.jl/stable/",
    "SparseMatrixColorings" => "https://juliadiff.org/SparseMatrixColorings.jl/stable/",
    "Symbolics" => "https://symbolics.juliasymbolics.org/stable/",
)

readme_str = read(joinpath(@__DIR__, "..", "README.md"), String)
readme_str = replace(readme_str, "> [!CAUTION]\n> " => "!!! warning\n    ")
#= The all-contributors table is raw HTML, which Documenter's Markdown parser escapes
instead of rendering. Wrap it in a `@raw html` block, leaving the README untouched. =#
readme_str = replace(
    readme_str,
    r"<!-- ALL-CONTRIBUTORS-LIST:START.*?<!-- ALL-CONTRIBUTORS-LIST:END -->"s =>
        table -> "```@raw html\n$table\n```",
)
# GitHub lowercases heading anchors, Documenter does not, so the badge link needs fixing.
readme_str = replace(readme_str, "](#contributors)" => "](#Contributors)")
write(joinpath(@__DIR__, "src", "index.md"), readme_str)

makedocs(;
    modules = [DifferentiationInterface],
    authors = "Guillaume Dalle, Adrian Hill",
    sitename = "DifferentiationInterface.jl",
    format = Documenter.HTML(; assets = ["assets/favicon.ico"]),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorials/basic.md", "tutorials/advanced.md"],
        "api.md",
        "Explanation" => [
            "explanation/arguments.md",
            "explanation/operators.md",
            "explanation/backends.md",
            "explanation/advanced.md",
        ],
        "FAQ" => ["faq/limitations.md", "faq/differentiability.md"],
        "Development" => [
            "dev/internals.md",
            "dev/math.md",
            "dev/contributing.md",
        ],
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
