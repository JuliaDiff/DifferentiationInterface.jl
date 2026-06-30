## Backend counterparts

# Pin the mode while preserving the function annotation type `A`.

function DI.forward_counterpart(::AutoEnzyme{M, A}) where {M, A}
    return AutoEnzyme(; mode = Forward, function_annotation = A)
end

function DI.reverse_counterpart(::AutoEnzyme{M, A}) where {M, A}
    return AutoEnzyme(; mode = Reverse, function_annotation = A)
end
