# DifferentiationInterfaceTest

| Category | Badges |
|---|---|
| Build status | [![Tests](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/JuliaDiff/DifferentiationInterface.jl/branch/main/graph/badge.svg?flag=DIT)](https://app.codecov.io/gh/JuliaDiff/DifferentiationInterface.jl) |
| Documentation | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/)     [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac) |
| Code quality | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) |
| References | [![DOI](https://zenodo.org/badge/740973714.svg)](https://zenodo.org/doi/10.5281/zenodo.11092033) |

Testing and benchmarking utilities for automatic differentiation (AD) in Julia, based on [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl/tree/main/DifferentiationInterface).

## Goal

Make it easy to know, for a given function:

- which AD backends can differentiate it
- how fast they can do it

## Features

- Definition of custom test scenarios
- Correctness tests
- Type stability tests
- Count calls to the function
- Benchmark runtime and allocations

## Installation

To install the stable version of the package, run the following code in a Julia REPL:

```julia
using Pkg

Pkg.add("DifferentiationInterfaceTest")
```

To install the development version, run this instead:

```julia
using Pkg

Pkg.add(;
    url="https://github.com/JuliaDiff/DifferentiationInterface.jl",
    subdir="DifferentiationInterface",
)

Pkg.add(;
    url="https://github.com/JuliaDiff/DifferentiationInterface.jl",
    subdir="DifferentiationInterfaceTest",
)
```

## Citation

See the citation instructions for [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl/tree/main/DifferentiationInterface).
