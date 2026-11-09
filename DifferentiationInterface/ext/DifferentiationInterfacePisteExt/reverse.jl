## Reverse mode: one recording of `f` on a `Piste.Tape`, one sweep per cotangent

const AutoPisteReverse{chunksize} = AutoPiste{ReverseMode, chunksize}

### Utilities

translate_rev(c::DI.Cache) = DI.recursive_similar(DI.unwrap(c), TapedReal)
translate_rev(c::DI.Context) = DI.unwrap(c)

function fix_contexts_rev(f::F, contexts::Vararg{DI.Context, C}) where {F, C}
    return DI.fix_tail(f, map(translate_rev, contexts)...)
end

# An in-place `f!(y, x)` is recorded through an out-of-place closure that
# writes into a tracked copy of `y`. Starting from detached zeros means an
# entry `f!` never writes is a constant, not uninitialized memory.
struct InPlaceAsOutOfPlace{F, Y}
    f!::F
    y::Y
end

function (g::InPlaceAsOutOfPlace)(x, contexts::Vararg{Any, C}) where {C}
    ytape = similar(g.y, TapedReal)
    fill!(ytape, zero(TapedReal))
    g.f!(ytape, x, contexts...)
    return ytape
end

workspace(x::Number) = ReverseWorkspace(1)
workspace(x::AbstractArray) = ReverseWorkspace(length(x))

vjp(fc::F, ws, x::Number, dy) where {F} = rev_value_and_pullback(fc, x, dy, ws)

function vjp(fc::F, ws, x::AbstractArray, dy) where {F}
    return rev_value_and_pullback!(similar(x, float(eltype(x))), fc, x, dy, ws)
end

vjp!(dx, fc::F, ws, x::AbstractArray, dy) where {F} = rev_value_and_pullback!(dx, fc, x, dy, ws)

# One recording per cotangent: the first also supplies the primal.
function value_and_vjps(fc::F, ws, x, ty::NTuple{B}) where {F, B}
    y, dx1 = vjp(fc, ws, x, first(ty))
    rest = map(dy -> last(vjp(fc, ws, x, dy)), Base.tail(ty))
    return y, (dx1, rest...)
end

function value_and_vjps!(tx::NTuple{B}, fc::F, ws, x, ty::NTuple{B}) where {F, B}
    y, _ = vjp!(first(tx), fc, ws, x, first(ty))
    for b in 2:B
        vjp!(tx[b], fc, ws, x, ty[b])
    end
    return y, tx
end

### Pullback, one argument

struct PistePullbackPrep{SIG} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    ws::ReverseWorkspace
end

function DI.prepare_pullback_nokwarg(
        strict::Val,
        f::F,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    return PistePullbackPrep(_sig, workspace(x))
end

function DI.value_and_pullback(
        f::F,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    fc = fix_contexts_rev(f, contexts...)
    return value_and_vjps(fc, prep.ws, x, ty)
end

function DI.value_and_pullback!(
        f::F,
        tx::NTuple,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    fc = fix_contexts_rev(f, contexts...)
    return value_and_vjps!(tx, fc, prep.ws, x, ty)
end

function DI.pullback(
        f::F,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return last(DI.value_and_pullback(f, prep, backend, x, ty, contexts...))
end

function DI.pullback!(
        f::F,
        tx::NTuple,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return last(DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...))
end

### Pullback, two arguments

function DI.prepare_pullback_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    _sig = DI.signature(f!, y, backend, x, ty, contexts...; strict)
    return PistePullbackPrep(_sig, workspace(x))
end

function DI.value_and_pullback(
        f!::F,
        y,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    fc = fix_contexts_rev(InPlaceAsOutOfPlace(f!, y), contexts...)
    yval, tx = value_and_vjps(fc, prep.ws, x, ty)
    copyto!(y, yval)
    return y, tx
end

function DI.value_and_pullback!(
        f!::F,
        y,
        tx::NTuple,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, ty, contexts...)
    fc = fix_contexts_rev(InPlaceAsOutOfPlace(f!, y), contexts...)
    yval, _ = value_and_vjps!(tx, fc, prep.ws, x, ty)
    copyto!(y, yval)
    return y, tx
end

function DI.pullback(
        f!::F,
        y,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return last(DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...))
end

function DI.pullback!(
        f!::F,
        y,
        tx::NTuple,
        prep::PistePullbackPrep,
        backend::AutoPisteReverse,
        x,
        ty::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return last(DI.value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...))
end

### Gradient: a single pullback with cotangent 1, reusing the workspace

struct PisteReverseGradientPrep{SIG} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    ws::ReverseWorkspace
end

function DI.prepare_gradient_nokwarg(
        strict::Val,
        f::F,
        backend::AutoPisteReverse,
        x::AbstractArray,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    return PisteReverseGradientPrep(_sig, workspace(x))
end

function DI.value_and_gradient!(
        f::F,
        grad,
        prep::PisteReverseGradientPrep,
        backend::AutoPisteReverse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = fix_contexts_rev(f, contexts...)
    return vjp!(grad, fc, prep.ws, x, 1.0)
end

function DI.gradient!(
        f::F,
        grad,
        prep::PisteReverseGradientPrep,
        backend::AutoPisteReverse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return last(DI.value_and_gradient!(f, grad, prep, backend, x, contexts...))
end

function DI.value_and_gradient(
        f::F,
        prep::PisteReverseGradientPrep,
        backend::AutoPisteReverse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    grad = similar(x, float(eltype(x)))
    return DI.value_and_gradient!(f, grad, prep, backend, x, contexts...)
end

function DI.gradient(
        f::F,
        prep::PisteReverseGradientPrep,
        backend::AutoPisteReverse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return last(DI.value_and_gradient(f, prep, backend, x, contexts...))
end
