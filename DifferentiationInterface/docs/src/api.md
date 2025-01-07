# API

```@meta
CollapsedDocStrings = true
```

```@docs
DifferentiationInterface
```

## Argument wrappers

```@docs
Context
Constant
Cache
```

## First order

### Pushforward

```@docs
prepare_pushforward
prepare_pushforward_same_point
pushforward
pushforward!
value_and_pushforward
value_and_pushforward!
```

### Pullback

```@docs
prepare_pullback
prepare_pullback_same_point
pullback
pullback!
value_and_pullback
value_and_pullback!
```

### Derivative

```@docs
prepare_derivative
derivative
derivative!
value_and_derivative
value_and_derivative!
```

### Gradient

```@docs
prepare_gradient
gradient
gradient!
value_and_gradient
value_and_gradient!
```

### Jacobian

```@docs
prepare_jacobian
jacobian
jacobian!
value_and_jacobian
value_and_jacobian!
MixedMode
```

## Second order

```@docs
SecondOrder
```

### Second derivative

```@docs
prepare_second_derivative
second_derivative
second_derivative!
value_derivative_and_second_derivative
value_derivative_and_second_derivative!
```

### Hessian-vector product

```@docs
prepare_hvp
prepare_hvp_same_point
hvp
hvp!
gradient_and_hvp
gradient_and_hvp!
```

### Hessian

```@docs
prepare_hessian
hessian
hessian!
value_gradient_and_hessian
value_gradient_and_hessian!
```

## Utilities

### Backend queries

```@docs
check_available
check_inplace
DifferentiationInterface.outer
DifferentiationInterface.inner
```

### Backend switch

```@docs
DifferentiateWith
```

### Sparsity detection

```@docs
DenseSparsityDetector
```

### Get Dualized array

```@docs
overloaded_inputs
```

## Internals

The following is not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Public = false
Filter = t -> !(Symbol(t) in [:outer, :inner])
```
