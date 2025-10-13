module DifferentiationInterfacePolyesterForwardDiffExt

using ADTypes: AutoForwardDiff, AutoPolyesterForwardDiff
import DifferentiationInterface as DI
using LinearAlgebra: mul!
using PolyesterForwardDiff: threaded_gradient!, threaded_jacobian!
using ForwardDiff: Chunk
using DiffResults: DiffResults

const FDExt = Base.get_extension(DI, :DifferentiationInterfaceForwardDiffExt)
@assert !isnothing(FDExt)

function single_threaded(backend::AutoPolyesterForwardDiff{chunksize, T}) where {chunksize, T}
    return AutoForwardDiff(; chunksize, tag = backend.tag)
end

DI.check_available(::AutoPolyesterForwardDiff) = true
DI.inner_preparation_behavior(::AutoPolyesterForwardDiff) = DI.PrepareInnerOverload()

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")
include("misc.jl")

end # module
