"""
    DifferentiationInterface

An experimental redesign for [AbstractDifferentiation.jl]
(https://github.com/JuliaDiff/AbstractDifferentiation.jl).

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using DocStringExtensions

abstract type AbstractBackend end

"""
    ChainRulesBackend{RC}

Performs autodiff with any package based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), like [Zygote.jl](https://github.com/FluxML/Zygote.jl) or [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl).

This must be constructed with an appropriate [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) instance:

```julia
using Zygote, DifferentiationInterface
backend = ChainRulesBackend(Zygote.ZygoteRuleConfig())
```
"""
struct ChainRulesBackend{RC} <: AbstractBackend
    ruleconfig::RC
end

Base.string(backend::ChainRulesBackend) = "ChainRulesBackend($(backend.ruleconfig))"

"""
    FiniteDiffBackend

Performs autodiff with [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
struct FiniteDiffBackend <: AbstractBackend end

"""
    EnzymeBackend

Performs autodiff with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
"""
struct EnzymeBackend <: AbstractBackend end

"""
    ForwardDiffBackend

Performs autodiff with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
"""
struct ForwardDiffBackend <: AbstractBackend end

"""
    ReverseDiffBackend

Performs autodiff with [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).
"""
struct ReverseDiffBackend <: AbstractBackend end

"""
    pushforward!(dy, backend, f, x, dx[, stuff])

Compute a Jacobian-vector product inside `dy` and return it.

# Arguments

- `dy`: cotangent, might be modified
- `backend`: autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dx`: tangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function pushforward! end

"""
    pullback!(dx, backend, f, x, dy[, stuff])

Compute a vector-Jacobian product inside `dx` and return it.

# Arguments

- `dx`: tangent, might be modified
- `backend`: autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function pullback! end

export ChainRulesBackend,
    EnzymeBackend, FiniteDiffBackend, ForwardDiffBackend, ReverseDiffBackend
export pushforward!, pullback!

end
