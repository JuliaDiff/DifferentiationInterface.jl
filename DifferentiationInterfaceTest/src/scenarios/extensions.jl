"""
    static_scenarios()

Create a vector of [`Scenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).

!!! warning

    This function requires StaticArrays.jl to be loaded (it is implemented in a package extension).
"""
function static_scenarios end

"""
    component_scenarios()

Create a vector of [`Scenario`](@ref)s with component array types from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl).

!!! warning

    This function requires ComponentArrays.jl to be loaded (it is implemented in a package extension).
"""
function component_scenarios end

"""
    gpu_scenarios()

Create a vector of [`Scenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).

!!! warning

    This function requires JLArrays.jl to be loaded (it is implemented in a package extension).
"""
function gpu_scenarios end

"""
    flux_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with neural networks from [Flux.jl](https://github.com/FluxML/Flux.jl).

!!! warning

    This function requires FiniteDifferences.jl and Flux.jl to be loaded (it is implemented in a package extension).

!!! danger

    These scenarios are still experimental and not part of the public API.
    Their ground truth values are computed with finite differences, and thus subject to imprecision.
"""
function flux_scenarios end

"""
    flux_isapprox(x, y; atol, rtol)

Approximate comparison function to use in correctness tests with gradients of Flux.jl networks.
"""
function flux_isapprox end

"""
    lux_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with neural networks from [Lux.jl](https://github.com/LuxDL/Lux.jl).

!!! warning

    This function requires ComponentArrays.jl, ForwardDiff.jl, Lux.jl and LuxTestUtils.jl to be loaded (it is implemented in a package extension).

!!! danger

    These scenarios are still experimental and not part of the public API.
"""
function lux_scenarios end

"""
    lux_isapprox(x, y; atol, rtol)

Approximate comparison function to use in correctness tests with gradients of Lux.jl networks.
"""
function lux_isapprox end
