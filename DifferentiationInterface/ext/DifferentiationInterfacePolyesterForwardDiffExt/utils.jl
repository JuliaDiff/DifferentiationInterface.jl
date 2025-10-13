function DI.pick_batchsize(backend::AutoPolyesterForwardDiff, x::AbstractArray)
    return DI.pick_batchsize(single_threaded(backend), x)
end

function DI.pick_batchsize(backend::AutoPolyesterForwardDiff, N::Integer)
    return DI.pick_batchsize(single_threaded(backend), N)
end

function DI.threshold_batchsize(
        backend::AutoPolyesterForwardDiff{chunksize1}, chunksize2::Integer
    ) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoPolyesterForwardDiff(; chunksize, tag = backend.tag)
end
