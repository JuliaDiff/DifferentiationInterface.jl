module DifferentiationInterfaceStaticArraysExt

using ADTypes: AutoForwardDiff, AutoEnzyme
import DifferentiationInterface as DI
using StaticArrays: SArray, Size, SOneTo, StaticArray, similar_type

# a flat vector of the same kind (e.g. `MVector` for an `MMatrix`) can be concatenated without heap allocations
DI.vec_block(a::StaticArray) = similar_type(a, Size(length(a)))(a)

function DI.stack_vec_col(t::NTuple{B, <:StaticArray}) where {B}
    return hcat(map(DI.vec_block, t)...)
end

function DI.stack_vec_row(t::NTuple{B, <:StaticArray}) where {B}
    return vcat(transpose.(map(DI.vec_block, t))...)
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
        block::F, ::DI.BatchSizeSettings{B, singlebatch, aligned}, ::NTuple{B, T}
    ) where {F, B, singlebatch, aligned, T <: StaticArray}
    N, A = static_batch_count(T, B)
    M = hcat(ntuple(block, Val(A))...)
    if aligned
        return M
    else
        return M[:, SOneTo(N)]
    end
end

function DI.stack_batches_row(
        block::F, ::DI.BatchSizeSettings{B, singlebatch, aligned}, ::NTuple{B, T}
    ) where {F, B, singlebatch, aligned, T <: StaticArray}
    N, A = static_batch_count(T, B)
    M = vcat(ntuple(block, Val(A))...)
    if aligned
        return M
    else
        return M[SOneTo(N), :]
    end
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
