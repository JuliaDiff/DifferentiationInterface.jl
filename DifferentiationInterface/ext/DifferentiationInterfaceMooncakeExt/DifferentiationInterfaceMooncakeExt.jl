module DifferentiationInterfaceMooncakeExt

using ADTypes: ADTypes, AutoMooncake, AutoMooncakeForward
import DifferentiationInterface as DI
using Mooncake:
    Mooncake,
    CoDual,
    Config,
    Dual,
    prepare_derivative_cache,
    prepare_gradient_cache,
    prepare_pullback_cache,
    primal,
    tangent,
    tangent_type,
    value_and_derivative!!,
    value_and_gradient!!,
    value_and_pullback!!,
    zero_dual,
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
    _copy_to_output!!,
    primal_to_tangent!!,
    tangent_to_primal!!

const AnyAutoMooncake{C} = Union{AutoMooncake{C}, AutoMooncakeForward{C}}

DI.check_available(::AnyAutoMooncake{C}) where {C} = true
DI.inner_preparation_behavior(::AutoMooncakeForward) = DI.PrepareInnerSimple()

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")
include("forward_onearg.jl")
include("forward_twoarg.jl")
include("differentiate_with.jl")

end
