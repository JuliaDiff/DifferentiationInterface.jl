"""
    AutoHyperHessians(; chunksize = nothing)

Lightweight ADTypes backend tag for HyperHessians. The `chunksize` keyword can
be set to a positive `Int` to override HyperHessians' chunk heuristic; `nothing`
lets HyperHessians choose.
"""
struct AutoHyperHessians{CS} <: ADTypes.AbstractADType
    chunksize::CS
    function AutoHyperHessians(; chunksize::Union{Nothing, Int} = nothing)
        if chunksize isa Int
            chunksize > 0 || throw(ArgumentError("chunksize must be positive, got $chunksize"))
        end
        return new{typeof(chunksize)}(chunksize)
    end
end
