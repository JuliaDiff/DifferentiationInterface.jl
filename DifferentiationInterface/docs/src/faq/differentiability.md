# Differentiability

DifferentiationInterface.jl and its sibling package [DifferentiationInterfaceTest.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl/tree/main/DifferentiationInterfaceTest) allow you to try out differentiation of existing code with a variety of AD backends.
However, they will not help you _write_ differentiable code in the first place.
To make your functions compatible with several backends, you need to mind the restrictions imposed by each one.

The list of backends available at [juliadiff.org](https://juliadiff.org/) is split into 2 main families: operator overloading and source transformation.
Writing differentiable code requires a specific approach in each paradigm:

- For operator overloading, ensure type-genericity.
- For source transformation, rely on existing rules or write your own.

!!! tip
    Depending on your intended use case, you may not need to ensure compatibility with every single backend.
    In particular, some applications strongly suggest a specific "mode" of AD (forward or reverse), in which case backends limited to the other mode are mostly irrelevant.

In what follows, we do not discuss AD with finite differences ([FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) and [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)) because those packages will work as long as your function itself can run, which is obviously a prerequisite.

## Operator overloading

### ForwardDiff

One of the most common backends in the ecosystem is [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
It performs AD at a scalar level by replacing plain numbers with [`Dual` numbers](https://juliadiff.org/ForwardDiff.jl/stable/dev/how_it_works/), which carry derivative information.
As explained in the [limitations of ForwardDiff](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/), this will only work if the differentiated code does not restrict number types too much.
Otherwise, you may encounter errors like this one:

```julia
MethodError: no method matching Float64(::ForwardDiff.Dual{...})
```

To prevent them, here are a few things to look out for:

- Avoid functions with overly specific type annotations.

```julia
f(x::Vector{Float64}) = ... # bad
f(x::AbstractVector{<:Real}) = ... # good
```

- When creating new containers or buffers, adapt to the input number type if necessary.

```julia
tmp = zeros(length(x))  # bad
tmp = zeros(eltype(x), length(x))  # good
tmp = similar(x)  # best when possible
```

In some situations, manually writing overloads for `x::Dual` or `x::AbstractArray{<:Dual}` can be necessary.

### ReverseDiff

[ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) relies on operator overloading for scalars, but also for arrays.
The relevant types are called `TrackedReal` and `TrackedArray`, they have a set of [limitations](https://juliadiff.org/ReverseDiff.jl/stable/limits/) very similar to that of ForwardDiff.jl's `Dual` and will cause similar errors. 

### Symbolic backends

[Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) and [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl) are also based on operator overloading.
However, their respective number types are a bit different because they represent symbolic variables instead of numerical values.
The operator overloading aims at reconstructing a symbolic representation of the function (an equation, more or less), which means certain language constructs will not be tolerated even though ForwardDiff.jl or ReverseDiff.jl could handle them.

## Source transformation

### Zygote

[Zygote.jl](https://github.com/FluxML/Zygote.jl) can differentiate a lot of Julia code, but it does have some major [limitations](https://fluxml.ai/Zygote.jl/stable/limitations/).
The most frequently encountered is the lack of support for mutation: if you try to modify the contents of an array during differentiation, you will get an error like

```julia
ERROR: Mutating arrays is not supported
```

Mutations and some other language constructs (exceptions, foreign calls) will make a function incompatible with Zygote.
In such cases, the proper workaround is to define a reverse rule (`rrule`) for that function using [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
You can find a [pedagogical example](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/example.html) for rule-writing in the documentation of ChainRulesCore.jl.

### Enzyme

By targeting a lower-level code representation than Zygote.jl, [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is able to differentiate a much wider set of functions.
The [FAQ](https://enzymead.github.io/Enzyme.jl/stable/faq/) gives some details on the breadth of coverage, but it should be enough for a lot of use cases.

Enzyme.jl also has an extensible [rule system](https://enzymead.github.io/Enzyme.jl/stable/generated/custom_rule/) which you can use to circumvent differentiation errors.
Note that its rule writing is very different from ChainRulesCore.jl due to the presence of input activity [annotations](https://enzymead.github.io/Enzyme.jl/stable/api/#EnzymeCore.Annotation).

### Mooncake

[Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) is a recent package which also handles a large subset of all Julia programs out-of-the-box.

Its [rule system](https://chalk-lab.github.io/Mooncake.jl/stable/understanding_mooncake/rule_system/) is less expressive than that of Enzyme.jl, which might make it easier to start with.

## A rule mayhem?

To summarize, here are the main rule systems which coexist at the moment:

- `Dual` numbers in ForwardDiff.jl
- ChainRulesCore.jl
- Enzyme.jl
- Mooncake.jl

### Rule translation

This split situation is unfortunate, but AD packages are so complex that making a cross-backend rule system is a very ambitious endeavor.
ChainRulesCore.jl is the closest thing we have to a standard, but it does not handle mutation.
As a result, Enzyme.jl and Mooncake.jl both rolled out their own designs, which are not mutually compatible.
There are, however, translation utilities:

- from ChainRulesCore.jl to ForwardDiff.jl with [ForwardDiffChainRules.jl](https://github.com/ThummeTo/ForwardDiffChainRules.jl)
- from ChainRulesCore.jl to Enzyme.jl with [`Enzyme.@import_rrule`](https://enzymead.github.io/Enzyme.jl/stable/api/#Enzyme.@import_rrule-Tuple)
- from ChainRulesCore.jl to Mooncake.jl with [`Mooncake.@from_rrule`](https://chalk-lab.github.io/Mooncake.jl/stable/utilities/defining_rules/#Using-ChainRules.jl)

### Backend switch

Also note the existence of [`DifferentiationInterface.DifferentiateWith`](@ref), which allows the user to wrap a function that should be differentiated with a specific backend.

Right now, it only targets [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), [Mooncake.jl](), [ChainRules.jl](https://juliadiff.org/ChainRulesCore.jl/stable/)-compatible backends (e.g., [Zygote.jl](https://github.com/FluxML/Zygote.jl)), but PRs are welcome to define Enzyme.jl rules for this object.