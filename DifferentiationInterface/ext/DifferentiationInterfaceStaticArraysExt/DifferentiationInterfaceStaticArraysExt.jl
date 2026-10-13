module DifferentiationInterfaceStaticArraysExt

using ADTypes: AutoForwardDiff, AutoEnzyme
import DifferentiationInterface as DI
using StaticArrays: SArray, SOneTo, StaticArray

function DI.stack_vec_col(t::NTuple{B, <:StaticArray}) where {B}
    return hcat(map(vec, t)...)
end

function DI.stack_vec_row(t::NTuple{B, <:StaticArray}) where {B}
    return vcat(transpose.(map(vec, t))...)
end

# The number of seeds `N` and of batches `A` are deduced from the seed type,
# so that concatenating the blocks preserves static sizes.

function static_batch_count(::Type{T}, B::Int) where {T <: StaticArray}
    N = length(T)
    A = cld(N, B)
    # otherwise `hcat()` would silently return an empty `Vector{Any}`
    A > 0 || throw(ArgumentError("Cannot concatenate zero batches"))
    return N, A
end

function DI.stack_batches_col(
        blocks::AbstractVector{<:StaticArray},
        ::DI.BatchSizeSettings{B, singlebatch, aligned},
        ::NTuple{B, T},
    ) where {B, singlebatch, aligned, T <: StaticArray}
    N, A = static_batch_count(T, B)
    M = hcat(ntuple(a -> blocks[a], Val(A))...)
    return aligned ? M : M[:, SOneTo(N)]
end

function DI.stack_batches_row(
        blocks::AbstractVector{<:StaticArray},
        ::DI.BatchSizeSettings{B, singlebatch, aligned},
        ::NTuple{B, T},
    ) where {B, singlebatch, aligned, T <: StaticArray}
    N, A = static_batch_count(T, B)
    M = vcat(ntuple(a -> blocks[a], Val(A))...)
    return aligned ? M : M[SOneTo(N), :]
end

DI.ismutable_array(::Type{<:SArray}) = false

function DI.pick_batchsize(::DI.AutoSimpleFiniteDiff{nothing}, x::StaticArray)
    return DI.BatchSizeSettings{length(x), true, true}(length(x))
end

function DI.pick_batchsize(::AutoForwardDiff{nothing}, x::StaticArray)
    return DI.BatchSizeSettings{length(x), true, true}(length(x))
end

function DI.pick_batchsize(::AutoEnzyme, x::StaticArray)
    return DI.BatchSizeSettings{length(x), true, true}(length(x))
end

function DI.pick_batchsize(
        ::DI.AutoSimpleFiniteDiff{chunksize}, x::StaticArray
    ) where {chunksize}
    return DI.BatchSizeSettings(chunksize, length(x))
end

function DI.pick_batchsize(::AutoForwardDiff{chunksize}, x::StaticArray) where {chunksize}
    return DI.BatchSizeSettings(chunksize, length(x))
end

end
