![DifferentiationInterface Logo](https://raw.githubusercontent.com/JuliaDiff/DifferentiationInterface.jl/main/DifferentiationInterface/docs/src/assets/logo.svg)

# DifferentiationInterface

[![Build Status](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDiff/DifferentiationInterface.jl/branch/main/graph/badge.svg?flag=DI)](https://app.codecov.io/gh/JuliaDiff/DifferentiationInterface.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/740973714.svg)](https://zenodo.org/doi/10.5281/zenodo.11092033)

|           Package            |                                                                                                                                                 Docs                                                                                                                                                 |
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
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
  - [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) (see below)
  - [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl)
  - [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)
  - [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
  - [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
  - [HyperHessians.jl](https://github.com/KristofferC/HyperHessians.jl)
  - [GTPSA.jl](https://github.com/bmad-sim/GTPSA.jl)
  - [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl)
  - [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl)
  - [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
  - [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
  - [Tracker.jl](https://github.com/FluxML/Tracker.jl)
  - [Zygote.jl](https://github.com/FluxML/Zygote.jl)

> [!CAUTION]
> Note that in some cases, going through DifferentiationInterface.jl might be slower or cause more errors than a direct call to the backend's API. This is especially true for Enzyme.jl, whose handling of activities and multiple arguments is not fully supported here. We are working on this challenge, and welcome any suggestions or contributions. Meanwhile, if differentiation fails or takes too long, consider using Enzyme.jl through its [native API](https://enzymead.github.io/Enzyme.jl/stable/) instead.

## Installation

To install the stable version of the package, run the following code in a Julia REPL:

```julia
using Pkg

Pkg.add("DifferentiationInterface")
```

To install the development version, run this instead:

```julia
using Pkg

Pkg.add(;
    url="https://github.com/JuliaDiff/DifferentiationInterface.jl",
    subdir="DifferentiationInterface",
)
```

## Example

```julia
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Zygote: Zygote  # AD backends you want to use

f(x) = sum(abs2, x)

x = [1.0, 2.0]

value_and_gradient(f, AutoForwardDiff(), x) # returns (5.0, [2.0, 4.0]) with ForwardDiff.jl
value_and_gradient(f, AutoEnzyme(), x) # returns (5.0, [2.0, 4.0]) with Enzyme.jl
value_and_gradient(f, AutoZygote(), x) # returns (5.0, [2.0, 4.0]) with Zygote.jl
```

To improve your performance by up to several orders of magnitude compared to this example, take a look at the tutorial and its section on operator preparation.

## Citation

Whenever you refer to this package or the ideas it contains, please cite:

 1. our preprint [*A Common Interface for Automatic Differentiation*](https://arxiv.org/abs/2505.05542);
 2. our inspiration [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl).

You can use the provided [`CITATION.cff`](https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/main/CITATION.cff) file or the following BibTeX entries:

```bibtex
@misc{dalle2025commoninterfaceautomaticdifferentiation,
      title={A Common Interface for Automatic Differentiation},
      author={Guillaume Dalle and Adrian Hill},
      year={2025},
      eprint={2505.05542},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2505.05542},
}

@misc{schäfer2022abstractdifferentiationjlbackendagnosticdifferentiableprogramming,
      title={AbstractDifferentiation.jl: Backend-Agnostic Differentiable Programming in Julia},
      author={Frank Schäfer and Mohamed Tarek and Lyndon White and Chris Rackauckas},
      year={2022},
      eprint={2109.12449},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2109.12449},
}
```

If you use the software, additionally cite us using the precise [Zenodo DOI](https://zenodo.org/records/11092033) of the package version you used, or the BibTeX entry below:

```bibtex
@software{dalleDifferentiationInterface2025,
      author={Dalle, Guillaume and Hill, Adrian},
      title={Differentiation{I}nterface.jl},
      year={2024},
      publisher={Zenodo},
      doi={10.5281/zenodo.11092033},
      url={https://doi.org/10.5281/zenodo.11092033},
}
```
