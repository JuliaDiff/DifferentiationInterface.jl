module DifferentiationInterfaceMooncakeExt

using ADTypes: ADTypes, AutoMooncake
import DifferentiationInterface as DI
using Mooncake:
    Mooncake,
    CoDual,
    Config,
    prepare_gradient_cache,
    prepare_pullback_cache,
    tangent_type,
    value_and_gradient!!,
    value_and_pullback!!,
    zero_tangent,
    rdata_type,
    fdata,
    rdata,
    tangent_type,
    NoTangent,
    @is_primitive,
    zero_fcodual,
    MinimalCtx,
    NoRData,
    primal,
    _copy_output,
    _copy_to_output!!

DI.check_available(::AutoMooncake) = true

get_config(::AutoMooncake{Nothing}) = Config()
get_config(backend::AutoMooncake{<:Config}) = backend.config

# tangents need to be copied before returning, otherwise they are still aliased in the cache
mycopy(x::Union{Number,AbstractArray{<:Number}}) = copy(x)
mycopy(x) = deepcopy(x)

include("onearg.jl")
include("twoarg.jl")
include("differentiate_with.jl")

end
