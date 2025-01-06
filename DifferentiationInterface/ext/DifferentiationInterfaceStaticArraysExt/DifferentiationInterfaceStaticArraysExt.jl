module DifferentiationInterfaceStaticArraysExt

using ADTypes: AutoForwardDiff, AutoEnzyme
import DifferentiationInterface as DI
using StaticArrays: SArray, StaticArray

function DI.stack_vec_col(t::NTuple{B,<:StaticArray}) where {B}
    return hcat(map(vec, t)...)
end

function DI.stack_vec_row(t::NTuple{B,<:StaticArray}) where {B}
    return vcat(transpose.(map(vec, t))...)
end

DI.ismutable_array(::Type{<:SArray}) = false

function DI.BatchSizeSettings(::DI.AutoSimpleFiniteDiff{nothing}, x::StaticArray)
    return DI.BatchSizeSettings{length(x),true,true}(length(x))
end

function DI.BatchSizeSettings(::AutoForwardDiff{nothing}, x::StaticArray)
    return DI.BatchSizeSettings{length(x),true,true}(length(x))
end

function DI.BatchSizeSettings(::AutoEnzyme, x::StaticArray)
    return DI.BatchSizeSettings{length(x),true,true}(length(x))
end

function DI.BatchSizeSettings(
    ::DI.AutoSimpleFiniteDiff{chunksize}, x::StaticArray
) where {chunksize}
    return DI.BatchSizeSettings{chunksize}(Val(length(x)))
end

function DI.BatchSizeSettings(
    ::AutoForwardDiff{chunksize}, x::StaticArray
) where {chunksize}
    return DI.BatchSizeSettings{chunksize}(Val(length(x)))
end

end
