# Limitations

## Multiple active arguments

At the moment, most backends cannot work with multiple active (differentiated) arguments.
As a result, DifferentiationInterface only supports a single active argument, called `x` in the documentation.

## Complex numbers

At the moment, complex numbers are only handled by a few AD backends, sometimes using different conventions.
As a result, DifferentiationInterface is only tested on real numbers and complex number support is not part of its API guarantees.
