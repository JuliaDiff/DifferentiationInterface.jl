stack_vec_col(t::NTuple) = stack(vec, t; dims=2)
stack_vec_row(t::NTuple) = stack(vec, t; dims=1)

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
function recursive_similar(x::Union{Tuple,NamedTuple}, ::Type{T}) where {T}
    return map(xi -> recursive_similar(xi, T), x)
end
