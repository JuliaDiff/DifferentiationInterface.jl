## Pushforward

struct PolyesterForwardDiffTwoArgPushforwardPrep{SIG,P} <: DI.PushforwardPrep{SIG}
    _sig::Type{SIG}
    single_threaded_prep::P
end

function DI.prepare_pushforward(
    f!,
    y,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {C}
    SIG = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    single_threaded_prep = DI.prepare_pushforward(
        f!, y, single_threaded(backend), x, tx, contexts...
    )
    return PolyesterForwardDiffTwoArgPushforwardPrep(SIG, single_threaded_prep)
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward(
        f!, y, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::PolyesterForwardDiffTwoArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward!(
        f!, y, ty, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    return DI.pushforward(
        f!, y, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward!(
    f!,
    y,
    ty::NTuple,
    prep::PolyesterForwardDiffTwoArgPushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    return DI.pushforward!(
        f!, y, ty, prep.single_threaded_prep, single_threaded(backend), x, tx, contexts...
    )
end

## Derivative

struct PolyesterForwardDiffTwoArgDerivativePrep{SIG,P} <: DI.DerivativePrep{SIG}
    _sig::Type{SIG}
    single_threaded_prep::P
end

function DI.prepare_derivative(
    f!,
    y,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {C}
    SIG = DI.signature(f!, y, backend, x, contexts...; strict)
    single_threaded_prep = DI.prepare_derivative(
        f!, y, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffTwoArgDerivativePrep(SIG, single_threaded_prep)
end

function DI.value_and_derivative(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    return DI.value_and_derivative(
        f!, y, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::PolyesterForwardDiffTwoArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    return DI.value_and_derivative!(
        f!, y, der, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.derivative(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    return DI.derivative(
        f!, y, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::PolyesterForwardDiffTwoArgDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    return DI.derivative!(
        f!, y, der, prep.single_threaded_prep, single_threaded(backend), x, contexts...
    )
end

## Jacobian

struct PolyesterForwardDiffTwoArgJacobianPrep{SIG,chunksize,P} <: DI.JacobianPrep{SIG}
    _sig::Type{SIG}
    chunk::Chunk{chunksize}
    single_threaded_prep::P
end

function DI.prepare_jacobian(
    f!,
    y,
    backend::AutoPolyesterForwardDiff{chunksize},
    x,
    contexts::Vararg{DI.Context,C};
    strict::Bool=false,
) where {chunksize,C}
    SIG = DI.signature(f!, y, backend, x, contexts...; strict)
    if isnothing(chunksize)
        chunk = Chunk(x)
    else
        chunk = Chunk{chunksize}()
    end
    single_threaded_prep = DI.prepare_jacobian(
        f!, y, single_threaded(backend), x, contexts...
    )
    return PolyesterForwardDiffTwoArgJacobianPrep(SIG, chunk, single_threaded_prep)
end

function DI.value_and_jacobian(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{DI.Context,C},
) where {K,C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        jac = similar(y, length(y), length(x))
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        fc!(y, x)
        return y, jac
    else
        return DI.value_and_jacobian(
            f!, y, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{DI.Context,C},
) where {K,C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        fc!(y, x)
        return y, jac
    else
        return DI.value_and_jacobian!(
            f!, y, jac, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.jacobian(
    f!,
    y,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        jac = similar(y, length(y), length(x))
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        return jac
    else
        return DI.jacobian(
            f!, y, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end

function DI.jacobian!(
    f!,
    y,
    jac,
    prep::PolyesterForwardDiffTwoArgJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{DI.Context,C},
) where {C}
    DI.check_prep(f!, y, prep, backend, x, contexts...)
    if contexts isa NTuple{C,DI.GeneralizedConstant}
        fc! = DI.with_contexts(f!, contexts...)
        threaded_jacobian!(fc!, y, jac, x, prep.chunk)
        return jac
    else
        return DI.jacobian!(
            f!, y, jac, prep.single_threaded_prep, single_threaded(backend), x, contexts...
        )
    end
end
