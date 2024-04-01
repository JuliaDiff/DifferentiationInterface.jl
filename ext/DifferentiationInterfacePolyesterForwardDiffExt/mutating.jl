## Pushforward

function DI.value_and_pushforward!!(
    f!, y, dy, backend::AnyAutoPolyForwardDiff, x, dx, extras::Nothing
)
    return DI.value_and_pushforward!!(f!, y, dy, single_threaded(backend), x, dx, extras)
end

function DI.value_and_derivative!!(
    f!, y, der, backend::AnyAutoPolyForwardDiff, x, extras::Nothing
)
    return DI.value_and_derivative!!(f!, y, der, single_threaded(backend), x, extras)
end

function DI.value_and_jacobian!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoPolyForwardDiff{C},
    x::AbstractArray,
    extras::Nothing,
) where {C}
    f!(y, x)
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return y, jac
end
