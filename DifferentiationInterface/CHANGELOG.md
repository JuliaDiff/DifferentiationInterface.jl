# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1]

### Feat

- Use Mooncake's internal copy utilities ([#809])

### Fixed

- Take `absstep` into account for FiniteDiff ([#812])
- Make basis work for `CuArray` ([#810])

## [0.7.0]

### Changed

- Preparation is now strict by default ([#799])
- New Arxiv preprint for citation ([#795])

## [0.6.54] - 2025-05-11

### Added

- Dependency compat bounds for extras ([#790])
- Error hints for Enzyme ([#788])

## [0.6.53] - 2025-05-07

### Changed

- Allocate Enzyme shadow memory during preparation ([#782])

[unreleased]: https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.1...main
[0.7.1]: https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.0...DifferentiationInterface-v0.7.1
[0.7.0]: https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.6.54...DifferentiationInterface-v0.7.0
[0.6.54]: https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.6.53...DifferentiationInterface-v0.6.54
[0.6.53]: https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.6.52...DifferentiationInterface-v0.6.53

[#812]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/812
[#810]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/810
[#809]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/809
[#799]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/799
[#795]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/795
[#790]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/790
[#788]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/788
[#782]: https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/782