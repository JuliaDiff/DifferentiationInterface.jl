package_name(b::AbstractADType) = package_name(typeof(b))

function package_name(::Type{B}) where {B<:AbstractADType}
    s = string(B)
    s = chopprefix(s, "ADTypes.")
    s = chopprefix(s, "Auto")
    k = findfirst('{', s)
    if isnothing(k)
        return s
    else
        return s[begin:(k - 1)]
    end
end

function package_name(::Type{SecondOrder{O,I}}) where {O,I}
    p1 = package_name(O)
    p2 = package_name(I)
    return p1 == p2 ? p1 : "$p1, $p2"
end

package_name(::Type{<:AutoSparse{D}}) where {D} = package_name(D)

function document_preparation(operator_name::AbstractString; same_point=false)
    if same_point
        return "To improve performance via operator preparation, refer to [`prepare_$(operator_name)`](@ref) and [`prepare_$(operator_name)_same_point`](@ref)."
    else
        return "To improve performance via operator preparation, refer to [`prepare_$(operator_name)`](@ref)."
    end
end
