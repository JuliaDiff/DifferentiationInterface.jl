![DifferentiationInterface Logo](https://raw.githubusercontent.com/JuliaDiff/DifferentiationInterface.jl/main/DifferentiationInterface/docs/src/assets/logo.svg)

# DifferentiationInterface

[![Build Status](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDiff/DifferentiationInterface.jl/branch/main/graph/badge.svg?flag=DI)](https://app.codecov.io/gh/JuliaDiff/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/740973714.svg)](https://zenodo.org/doi/10.5281/zenodo.11092033)

|           Package            |                                                                                                                                                    Docs                                                                                                                                                    |
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   DifferentiationInterface   |   [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/)     [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/dev/)   |
| DifferentiationInterfaceTest | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/) |

An interface to various automatic differentiation (AD) backends in Julia.

## Goal

This package provides a unified syntax to differentiate functions, including:

- First- and second-order operators (gradients, Jacobians, Hessians and more)
- In-place and out-of-place differentiation
- Preparation mechanism (e.g. to pre-allocate a cache or record a tape)
- Built-in sparsity handling
- Thorough validation on standard inputs and outputs (numbers, vectors, matrices)
- Testing and benchmarking utilities accessible to users with [DifferentiationInterfaceTest](https://github.com/JuliaDiff/DifferentiationInterface.jl/tree/main/DifferentiationInterfaceTest)

## Compatibility

We support the following backends defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl) (currently broken)
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
- [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl)
- [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)
- [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- [Mooncake.jl](https://github.com/compintell/Mooncake.jl)
- [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl)
- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
- [Tracker.jl](https://github.com/FluxML/Tracker.jl)
- [Zygote.jl](https://github.com/FluxML/Zygote.jl)

Note that in some cases, going through DifferentiationInterface.jl might be slower than a direct call to the backend's API.
This is mostly true for Enzyme.jl, whose handling of activities and multiple arguments unlocks additional performance.
We are working on this challenge, and welcome any suggestions or contributions.
Meanwhile, if differentiation fails or takes too long, consider using Enzyme.jl directly.

## Installation

To install the stable version of the package, run the following code in a Julia REPL:

```julia
using Pkg

Pkg.add("DifferentiationInterface")
```

To install the development version, run this instead:

```julia
using Pkg

Pkg.add(
    url="https://github.com/JuliaDiff/DifferentiationInterface.jl",
    subdir="DifferentiationInterface"
)
```

## Example

```julia
using DifferentiationInterface
import ForwardDiff, Enzyme, Zygote  # AD backends you want to use 

f(x) = sum(abs2, x)

x = [1.0, 2.0]

value_and_gradient(f, AutoForwardDiff(), x) # returns (5.0, [2.0, 4.0]) with ForwardDiff.jl
value_and_gradient(f, AutoEnzyme(),      x) # returns (5.0, [2.0, 4.0]) with Enzyme.jl
value_and_gradient(f, AutoZygote(),      x) # returns (5.0, [2.0, 4.0]) with Zygote.jl
```

To improve your performance by up to several orders of magnitude compared to this example, take a look at the tutorial and its section on operator preparation.

## Citation

Please cite both DifferentiationInterface.jl and its inspiration [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl), using the provided `CITATION.cff` file.
