function DI.overloaded_input(
        ::typeof(DI.pushforward), f::F, backend::AutoPolyesterForwardDiff, x, tx::NTuple{B}
    ) where {F, B}
    return DI.overloaded_input(DI.pushforward, f, single_threaded(backend), x, tx)
end

function DI.overloaded_input(
        ::typeof(DI.pushforward), f!::F, y, backend::AutoPolyesterForwardDiff, x, tx::NTuple{B}
    ) where {F, B}
    return DI.overloaded_input(DI.pushforward, f!, y, single_threaded(backend), x, tx)
end
