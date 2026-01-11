"""
!!! tip "DI-specific information"
    This part of the docstring is related to the use of `AutoReactant` inside [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl), or DI for short.
    Reactant's tutorial on [partial evaluation](https://enzymead.github.io/Reactant.jl/stable/tutorials/partial-evaluation) is useful reading to understand what follows.

The `AutoReactant` backend inside DI imposes the following restrictions / assumptions:

- The only supported operator (at the moment) is `DI.gradient` (along with its variants).
- The input `x` must be an `AbstractArray` such that `Reactant.ConcreteRArray(x)` is well-defined.
- By default, contexts such as `DI.Constant` and `DI.Cache` will be partially evaluated inside the compiled differentiation operator at preparation time. This means that the context value provided at preparation will be reused at every subsequent execution, while the context value provided at execution will be ignored. In particular, `DI.Cache` contexts will usually error and `DI.Constant` contexts will be frozen to one value.

To disable partial evaluation and enforce tracing of contexts instead, first wrap them into types that _you own_.
Then, overload [`DifferentiationInterface.to_reactant`](@ref) on these types to perform tracing in the way you see fit, for instance with `Reactant.to_rarray`.
Every value you choose not to trace will still be partially evaluated at preparation time.

# Example

```jldoctest
using DifferentiationInterface
import DifferentiationInterface as DI
import Reactant

struct MyArgument{T1 <: Number, T2 <: AbstractArray}
    u::T1
    v::T2
end

f(x, a::MyArgument) = a.u * sum(a.v .* x .^ 2)

DI.to_reactant(a::MyArgument) = Reactant.to_rarray(a; track_numbers = false)

# preparation time
x0 = zeros(2)
a0 = MyArgument(1.0, [2.0, 3.0])

# execution time
x = [4.0, 5.0]
a = MyArgument(6.0, [7.0, 8.0])

backend = AutoReactant()
prep = prepare_gradient(f, backend, x0, Constant(a0));

g = gradient(f, prep, backend, x, Constant(a))
g ≈ a0.u * 2 * (a.v .* x)  # a0.u is partially evaluated, a0.v is traced

# output

true
```
"""
AutoReactant

"""
    to_reactant(a)

Convert an argument `a` to an object `ar` containing the same values, where all the fields and subfields that can contain active (differentiated) data have been translated to [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) types such as [`ConcreteRArray`](@extref Reactant.ConcreteRArray) or [`ConcreteRNumber`](@extref Reactant.ConcreteRNumber).

!!! danger
    DifferentiationInterface.jl implements this function as the identity, on purpose.
    It should not be overloaded on base types, but only on types that you own, to modify the default behavior of `AutoReactant`.

# Example

```jldoctest
import DifferentiationInterface as DI
import Reactant

struct MyArgument{T1 <: Number, T2 <: AbstractArray}
    u::T1
    v::T2
end

DI.to_reactant(a::MyArgument) = Reactant.to_rarray(a; track_numbers = false)

a = MyArgument(1.0, [2.0, 3.0])
ar = DI.to_reactant(a)
ar isa MyArgument{Float64, <:Reactant.ConcreteRArray}

# output

true
```
"""
to_reactant(x) = x
