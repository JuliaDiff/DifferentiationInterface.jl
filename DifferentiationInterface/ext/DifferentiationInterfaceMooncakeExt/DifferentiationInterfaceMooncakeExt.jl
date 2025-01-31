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
    zero_tangent,
    prepare_pullback_cache,
    Mooncake

DI.check_available(::AutoMooncake) = true
DI.check_operator_overloading(::AutoMooncake) = true

copyto!!(dst::Number, src::Number) = convert(typeof(dst), src)
copyto!!(dst, src) = DI.ismutable_array(dst) ? copyto!(dst, src) : convert(typeof(dst), src)

get_config(::AutoMooncake{Nothing}) = Config()
get_config(backend::AutoMooncake{<:Config}) = backend.config

include("onearg.jl")
include("twoarg.jl")

end
