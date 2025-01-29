# Limitations

## Multiple active arguments

At the moment, most backends cannot work with multiple active (differentiated) arguments.
As a result, DifferentiationInterface only supports a single active argument, called `x` in the documentation.

## Complex numbers

Complex derivatives are only handled by a few AD backends, sometimes using different conventions.
To find the easiest common ground, DifferentiationInterface assumes that whenever complex numbers are involved, the function to differentiate is holomorphic.
This functionality is still considered experimental and not yet part of the public API guarantees.
If you work with non-holomorphic functions, you will need to manually separate real and imaginary parts.
