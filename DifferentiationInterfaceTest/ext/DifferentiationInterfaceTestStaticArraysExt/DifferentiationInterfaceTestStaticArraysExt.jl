module DifferentiationInterfaceTestStaticArraysExt

import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, spdiagm
using StaticArrays: MArray, MMatrix, MVector, SArray, SMatrix, SVector

static_num_to_vec(x::Number) = sin.(SVector(1, 2) .* x)
static_num_to_mat(x::Number) = hcat(static_num_to_vec(x), static_num_to_vec(3x))

const NTV                         = typeof(DIT.num_to_vec)
const NTM                         = typeof(DIT.num_to_mat)
mystatic(f::Function)             = f
mystatic(::NTV)                   = static_num_to_vec
mystatic(::NTM)                   = static_num_to_mat
mystatic(f::DIT.FunctionModifier) = f

mystatic(x::Number) = x

mystatic(x::AbstractVector{T}) where {T} = convert(SVector{length(x),T}, x)
mymutablestatic(x::AbstractVector{T}) where {T} = convert(MVector{length(x),T}, x)

function mystatic(x::AbstractMatrix{T}) where {T}
    return convert(SMatrix{size(x, 1),size(x, 2),T,length(x)}, x)
end
function mymutablestatic(x::AbstractMatrix{T}) where {T}
    return convert(MMatrix{size(x, 1),size(x, 2),T,length(x)}, x)
end

mystatic(x::Tuple) = map(mystatic, x)
mystatic(x::DI.Constant) = DI.Constant(mystatic(DI.unwrap(x)))
mystatic(x::DI.Cache{<:AbstractArray}) = DI.Cache(mymutablestatic(DI.unwrap(x)))
function mystatic(x::DI.Cache{<:Union{Tuple,NamedTuple}})
    return map(mystatic, map(DI.Cache, DI.unwrap(x)))
end
mystatic(::Nothing) = nothing

function mystatic(scen::DIT.Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, tang, contexts, res1, res2) = scen
    return DIT.Scenario{op,pl_op,pl_fun}(
        mystatic(f);
        x=mystatic(x),
        y=pl_fun == :in ? mymutablestatic(y) : mystatic(y),
        tang=mystatic(tang),
        contexts=mystatic(contexts),
        res1=mystatic(res1),
        res2=mystatic(res2),
    )
end

function DIT.static_scenarios(args...; kwargs...)
    scens = DIT.default_scenarios(args...; kwargs...)
    return mystatic.(scens)
end

end
