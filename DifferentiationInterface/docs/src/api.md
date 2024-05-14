```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API reference

```@docs
DifferentiationInterface
```

## Derivative

```@docs
prepare_derivative
derivative
derivative!
value_and_derivative
value_and_derivative!
```

## Gradient

```@docs
prepare_gradient
gradient
gradient!
value_and_gradient
value_and_gradient!
```

## Jacobian

```@docs
prepare_jacobian
jacobian
jacobian!
value_and_jacobian
value_and_jacobian!
```

## Second order

```@docs
SecondOrder
```

```@docs
prepare_second_derivative
second_derivative
second_derivative!
```

```@docs
prepare_hvp
prepare_hvp_same_point
hvp
hvp!
```

```@docs
prepare_hessian
hessian
hessian!
```

## Primitives

```@docs
prepare_pushforward
prepare_pushforward_same_point
pushforward
pushforward!
value_and_pushforward
value_and_pushforward!
```

```@docs
prepare_pullback
prepare_pullback_same_point
pullback
pullback!
value_and_pullback
value_and_pullback!
```

## Backend queries

```@docs
check_available
check_twoarg
check_hessian
```

## Translation

```@docs
DifferentiateWith
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Public = false
```
