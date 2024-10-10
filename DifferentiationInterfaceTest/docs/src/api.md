# API reference

```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

```@docs
DifferentiationInterfaceTest
```

## Entry points

```@docs
Scenario
test_differentiation
benchmark_differentiation
FIRST_ORDER
SECOND_ORDER
```

## Utilities

```@docs
DifferentiationBenchmarkDataRow
```

## Pre-made scenario lists

The precise contents of the scenario lists are not part of the API, only their existence.

```@docs
default_scenarios
sparse_scenarios
component_scenarios
gpu_scenarios
static_scenarios
```

## Utilities

```@docs
DifferentiationBenchmarkDataRow
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterfaceTest]
Public = false
```
