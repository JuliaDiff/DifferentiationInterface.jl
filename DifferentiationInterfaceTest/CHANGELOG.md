# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.2...main)

### Fixed

  - Allow JET v0.10 but disable test stability

## [0.10.2](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.1...DifferentiationInterfaceTest-v0.10.2)

### Fixed

  - Refactor test loops ([#848](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/848))

## [0.10.1](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.10.0...DifferentiationInterfaceTest-v0.10.1)

### Added

  - Improve support for empty inputs (still not guaranteed) ([#835](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/835))
  - Compute Scenario results with a reference backend ([#839](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/839))

### Fixed

  - Put test deps into `test/Project.toml` ([#840](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/840))
  - Set up `pre-commit` ([#837](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/837))
  - Bump compat for SparseConnectivityTracer v1 ([#823](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/823))

## [0.10.0](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.9.6...DifferentiationInterfaceTest-v0.10.0)

### Changed

  - Specify preparation arguments in DIT Scenario ([#786](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/786))

### Removed

  - Remove scenario lists from public API ([#796](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/796))

## [0.9.6](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterfaceTest-v0.9.5...DifferentiationInterfaceTest-v0.9.6) - 2025-03-28

### Added

  - Add new ConstantOrCache context ([#749](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/749))
  - Support nested tuples of arrays as Caches ([#748](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/748))
  - Test type consistency between preparation and execution ([#745](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/745))
