# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.13...main)

## [0.7.13](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.12...DifferentiationInterface-v0.7.13)

### Added

- Allow non-numeric arguments with AutoSymbolics ([#945](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/945))

### Fixed

- Function shadows for higher-order Enzyme ([#943](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/943))
- Improve wrong-mode pushforward/pullback ([#932](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/932), [#931](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/931))
- Clean up CI ([#926](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/926), [#924](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/924))

## [0.7.12](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.11...DifferentiationInterface-v0.7.12)

### Added

- Better documentation on argument assumptions ([#917](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/917))

### Fixed

- Speed up Mooncake in forward mode by preallocating tangents ([#915](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/915))
- Speed up Mooncake reverse mode with selective zeroing ([#916](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/916))  

## [0.7.11](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.10...DifferentiationInterface-v0.7.11)

### Fixed

- Allow (empty) sparse Hessians of linear functions ([#906](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/906))

## [0.7.10](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.9...DifferentiationInterface-v0.7.10)

### Fixed

- Add `Prep` to the public interface ([#875](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/875))
- Improve performance of unbatched out-of-place Jacobian ([#876](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/875))

## [0.7.9](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.8...DifferentiationInterface-v0.7.9)

### Fixed

- Handle empty row or column colors in mixed mode sparse Jacobian ([#864](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/864))

## [0.7.8](https://github.com/JuliaDiff/DifferentiationInterface.jl/compare/DifferentiationInterface-v0.7.7...DifferentiationInterface-v0.7.8)

### Added

- Support the new `ADTypes.NoAutoDiff` ([#851](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/851))

### Fixed

- Speed up Mooncake by avoiding tuple broadcasting ([#853](https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/853))

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
