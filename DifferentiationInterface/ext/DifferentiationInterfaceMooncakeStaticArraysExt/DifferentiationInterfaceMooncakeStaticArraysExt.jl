module DifferentiationInterfaceMooncakeStaticArraysExt

using Base: IEEEFloat
using DifferentiationInterface: DifferentiationInterface
using Mooncake: Mooncake
using StaticArrays: MArray, SArray

# Reach into the binary MooncakeExt to extend its `_to_friendly_value` hook.
# Both Mooncake and StaticArrays are loaded whenever this extension is active,
# so the binary extension is guaranteed to be loaded as well.
const _MooncakeExt = Base.get_extension(
    DifferentiationInterface, :DifferentiationInterfaceMooncakeExt,
)

# Restrict to scalar float / complex-float eltypes: those are the layouts where
# Mooncake's framework sends `SArray` / `MArray` through `AsRaw` and the single
# `data::NTuple` element-to-position mapping is 1:1 (no aliasing). Non-float
# eltypes hit Mooncake's element-wise `AbstractArray` recursion at
# `Mooncake/src/tangents/tangents.jl:1453`; let that path run unimpeded.
const _StaticEltype = Union{IEEEFloat, Complex{<:IEEEFloat}}

# Mooncake's `friendly_tangent_cache` framework defaults to `:as_raw` for
# `SArray` / `MArray` primals with float eltype, leaking a raw `Tangent` /
# `MutableTangent` instead of a primal-shaped value. Bridge that gap here. The
# reconstruction is unambiguous because the `data::NTuple` field maps each
# element to one logical position (unlike `Symmetric` / `Hermitian`, where a
# single stored entry can represent two positions).
@inline _MooncakeExt._to_friendly_value(
    t::Mooncake.Tangent, x::SArray{S, T}
) where {S, T <: _StaticEltype} = typeof(x)(Mooncake.val(t.fields.data))

@inline _MooncakeExt._to_friendly_value(
    t::Mooncake.MutableTangent, x::MArray{S, T}
) where {S, T <: _StaticEltype} = typeof(x)(Mooncake.val(t.fields.data))

end # module
