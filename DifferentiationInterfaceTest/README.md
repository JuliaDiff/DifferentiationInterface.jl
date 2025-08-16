# DifferentiationInterfaceTest

[![Build Status](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDiff/DifferentiationInterface.jl/branch/main/graph/badge.svg?flag=DIT)](https://app.codecov.io/gh/JuliaDiff/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/740973714.svg)](https://zenodo.org/doi/10.5281/zenodo.11092033)

| Package                      | Docs                                                                                                                                                                                                                                                                                                 |
|:----------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| DifferentiationInterface     | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/)     [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/dev/)     |
| DifferentiationInterfaceTest | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/) |

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
