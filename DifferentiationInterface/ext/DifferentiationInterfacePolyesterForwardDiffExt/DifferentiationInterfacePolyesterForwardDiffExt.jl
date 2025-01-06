module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
import DifferentiationInterface as DI
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using PolyesterForwardDiff.ForwardDiff: Chunk
using PolyesterForwardDiff.ForwardDiff.DiffResults: DiffResults

function single_threaded(backend::AutoPolyesterForwardDiff{chunksize,T}) where {chunksize,T}
    return AutoForwardDiff(; chunksize, tag=backend.tag)
end

DI.check_available(::AutoPolyesterForwardDiff) = true

function DI.BatchSizeSettings(backend::AutoPolyesterForwardDiff, x::AbstractArray)
    return DI.BatchSizeSettings(single_threaded(backend), x)
end

function DI.BatchSizeSettings(backend::AutoPolyesterForwardDiff, N::Integer)
    return DI.BatchSizeSettings(single_threaded(backend), N)
end

function DI.threshold_batchsize(
    backend::AutoPolyesterForwardDiff{chunksize1}, chunksize2::Integer
) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoPolyesterForwardDiff(; chunksize, tag=backend.tag)
end

include("onearg.jl")
include("twoarg.jl")

end # module
