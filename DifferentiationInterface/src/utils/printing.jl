required_packages(b::AbstractADType) = required_packages(typeof(b))

function required_packages(::Type{B}) where {B<:AbstractADType}
    s = string(B)
    s = chopprefix(s, "ADTypes.")
    s = chopprefix(s, "Auto")
    k = findfirst('{', s)
    if isnothing(k)
        return [s]
    else
        return [s[begin:(k - 1)]]
    end
end

function required_packages(::Type{SecondOrder{O,I}}) where {O,I}
    p1 = required_packages(O)
    p2 = required_packages(I)
    return unique(vcat(p1, p2))
end

function required_packages(::Type{MixedMode{F,R}}) where {F,R}
    p1 = required_packages(F)
    p2 = required_packages(R)
    return unique(vcat(p1, p2))
end

function required_packages(::Type{<:AutoSparse{D}}) where {D}
    return unique(vcat(required_packages(D), "SparseMatrixColorings"))
end

function document_preparation(operator_name::AbstractString; same_point=false)
    if same_point
        return "To improve performance via operator preparation, refer to [`prepare_$(operator_name)`](@ref) and [`prepare_$(operator_name)_same_point`](@ref)."
    else
        return "To improve performance via operator preparation, refer to [`prepare_$(operator_name)`](@ref)."
    end
end
