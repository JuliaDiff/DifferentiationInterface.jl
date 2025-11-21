stack_vec_col(t::NTuple) = stack(vec, t; dims = 2)
stack_vec_row(t::NTuple) = stack(vec, t; dims = 1)

"""
    ismutable_array(x)

Check whether `x` is a mutable array and return a `Bool`.

At the moment, this only returns `false` for `StaticArrays.SArray`.
"""
ismutable_array(::Type) = true
ismutable_array(x) = ismutable_array(typeof(x))

"""
    recursive_similar(x, T)

Apply `similar(_, T)` recursively to `x` or its components.

Works if `x` is an `AbstractArray` or a (nested) `NTuple` / `NamedTuple` of `AbstractArray`s.
"""
recursive_similar(x::AbstractArray, ::Type{T}) where {T} = similar(x, T)
function recursive_similar(x::Union{Tuple, NamedTuple}, ::Type{T}) where {T}
    return map(xi -> recursive_similar(xi, T), x)
end

"""
    get_pattern(M::AbstractMatrix)

Return the `Bool`-valued sparsity pattern for a given matrix.

Only specialized on `SparseMatrixCSC` because it is used with symbolic backends, and at the moment their sparse Jacobian/Hessian utilities return a `SparseMatrixCSC`.

The trivial dense fallback is designed to protect against a change of format in these packages.
"""
get_pattern(M::AbstractMatrix) = trues(size(M))

onlysecond((a, b)) = (a, only(b))

arroftup_to_tupofarr(x::NTuple) = x
arroftup_to_tupofarr(x::AbstractArray{<:NTuple{B}}) where {B} = ntuple(b -> getindex.(x, b), Val(B))
