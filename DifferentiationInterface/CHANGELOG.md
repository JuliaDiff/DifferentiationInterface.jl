# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.7...main)

### Fixed

  - Handle constant derivatives with runtime activity for Enzyme

## [0.7.7](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.6...DifferentiationInterface-v0.7.7)

### Fixed

  - Improve support for empty inputs (still not guaranteed) ([#835](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/835))

## [0.7.6](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.5...DifferentiationInterface-v0.7.6)

### Fixed

  - Put test deps into `test/Project.toml` ([#840](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/840))
  - Set up `pre-commit` ([#837](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/837))

### Fixed

  - Put test deps into `test/Project.toml` ([#840](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/840))

## [0.7.5](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.4...DifferentiationInterface-v0.7.5)

### Added

  - Support forward-mode Mooncake with `AutoMooncakeForward` ([#813](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/813))

## [0.7.4](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.3...DifferentiationInterface-v0.7.4)

### Added

  - Make `AutoForwardFromPrimitive` and `AutoReverseFromPrimitive` public ([#825](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/825))

### Fixed

  - Replace `one` with `oneunit` in basis computation ([#826](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/826))

## [0.7.3](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.2...DifferentiationInterface-v0.7.3)

### Fixed

  - Bump compat for SparseConnectivityTracer v1 ([#823](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/823))

## [0.7.2](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.1...DifferentiationInterface-v0.7.2)

### Feat

  - Backend switching for Mooncake ([#768](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/768))

### Fixed

  - Speed up sparse preparation for GPU arrays ([#818](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/818))

## [0.7.1](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.0...DifferentiationInterface-v0.7.1)

### Feat

  - Use Mooncake's internal copy utilities ([#809](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/809))

### Fixed

  - Take `absstep` into account for FiniteDiff ([#812](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/812))
  - Make basis work for `CuArray` ([#810](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/810))

## [0.7.0](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.6.54...DifferentiationInterface-v0.7.0)

### Changed

  - Preparation is now strict by default ([#799](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/799))
  - New Arxiv preprint for citation ([#795](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/795))

## [0.6.54](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.6.53...DifferentiationInterface-v0.6.54) - 2025-05-11

### Added

  - Dependency compat bounds for extras ([#790](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/790))
  - Error hints for Enzyme ([#788](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/788))

## [0.6.53](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.6.52...DifferentiationInterface-v0.6.53) - 2025-05-07

### Changed

  - Allocate Enzyme shadow memory during preparation ([#782](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/782))
