module DifferentiationInterfaceMooncakeExt

using ADTypes: ADTypes, AutoMooncake
import DifferentiationInterface as DI
using Mooncake:
    CoDual,
    Config,
    primal,
    tangent,
    tangent_type,
    value_and_pullback!!,
    value_and_gradient!!,
    zero_tangent,
    prepare_pullback_cache,
    Mooncake

DI.check_available(::AutoMooncake) = true

copyto!!(dst::Number, src::Number) = convert(typeof(dst), src)
copyto!!(dst, src) = DI.ismutable_array(dst) ? copyto!(dst, src) : convert(typeof(dst), src)

get_config(::AutoMooncake{Nothing}) = Config()
get_config(backend::AutoMooncake{<:Config}) = backend.config

# tangents need to be copied before returning, otherwise they are still aliased in the cache
mycopy(x::Union{Number,AbstractArray}) = copy(x)
mycopy(x) = deepcopy(x)

include("onearg.jl")
include("twoarg.jl")

end
