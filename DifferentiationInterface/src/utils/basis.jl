"""
    basis(a::AbstractArray, i)

Construct the `i`-th standard basis array in the vector space of `a`.
"""
function basis(a::AbstractArray{T}, i) where {T}
    b = similar(a)
    fill!(b, zero(T))
    b[i] = oneunit(T)
    if ismutable_array(a)
        return b
    else
        return map(+, zero(a), b)
    end
end

"""
    multibasis(a::AbstractArray, inds)

Construct the sum of the `i`-th standard basis arrays in the vector space of `a` for all `i ∈ inds`.
"""
function multibasis(a::AbstractArray{T}, inds) where {T}
    b = similar(a)
    fill!(b, zero(T))
    for i in inds
        b[i] = oneunit(T)
    end
    return ismutable_array(a) ? b : map(+, zero(a), b)
end
