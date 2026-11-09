## Forward mode: `Piste.Dual{B,T}` carries `B` directional derivatives at once

const AutoPisteForward{chunksize} = AutoPiste{ForwardMode, chunksize}

function DI.pick_batchsize(::AutoPisteForward{nothing}, N::Integer)
    return DI.BatchSizeSettings(DI.reasonable_batchsize(N, 8), N)
end

function DI.pick_batchsize(::AutoPisteForward{chunksize}, N::Integer) where {chunksize}
    return DI.BatchSizeSettings(chunksize, N)
end

function DI.threshold_batchsize(
        ::AutoPisteForward{chunksize1}, chunksize2::Integer
    ) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoPiste(; mode = ForwardMode(), chunksize)
end

### Utilities

dual_eltype(x) = float(eltype(x))

make_dual(::Type{T}, x::Number, tx::NTuple{B}) where {T, B} = Dual{B, T}(T(x), map(T, tx))

function make_dual(::Type{T}, x::AbstractArray, tx::NTuple{B}) where {T, B}
    return map((xi, txi...) -> Dual{B, T}(T(xi), map(T, txi)), x, tx...)
end

partial(y::Dual, b::Integer) = Piste.partials(y, b)
partial(y::Number, ::Integer) = zero(y)  # an output that does not depend on `x`

mypartials(::Val{B}, ydual::Number) where {B} = ntuple(b -> partial(ydual, b), Val(B))

function mypartials(::Val{B}, ydual::AbstractArray) where {B}
    return ntuple(b -> map(yi -> partial(yi, b), ydual), Val(B))
end

function mypartials!(ty::NTuple{B}, ydual::AbstractArray) where {B}
    for b in 1:B
        map!(yi -> partial(yi, b), ty[b], ydual)
    end
    return ty
end

myvalue(ydual::Number) = value(ydual)
myvalue(ydual::AbstractArray) = map(value, ydual)
myvalue!(y::AbstractArray, ydual::AbstractArray) = map!(value, y, ydual)

translate(::Type{D}, c::DI.Cache) where {D} = DI.recursive_similar(DI.unwrap(c), D)
translate(::Type{D}, c::DI.Context) where {D} = DI.unwrap(c)

function translate(::Type{D}, contexts::NTuple{C, DI.Context}) where {D, C}
    return map(c -> translate(D, c), contexts)
end

### Pushforward, one argument

struct PisteOneArgPushforwardPrep{SIG} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
end

function DI.prepare_pushforward_nokwarg(
        strict::Val,
        f::F,
        backend::AutoPisteForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    return PisteOneArgPushforwardPrep(_sig)
end

function compute_ydual(
        f::F, x, tx::NTuple{B}, contexts::Vararg{DI.Context, C}
    ) where {F, B, C}
    T = dual_eltype(x)
    xdual = make_dual(T, x, tx)
    contexts_dual = translate(Dual{B, T}, contexts)
    return f(xdual, contexts_dual...)
end

function DI.value_and_pushforward(
        f::F,
        prep::PisteOneArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ydual = compute_ydual(f, x, tx, contexts...)
    return myvalue(ydual), mypartials(Val(B), ydual)
end

function DI.value_and_pushforward!(
        f::F,
        ty::NTuple,
        prep::PisteOneArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ydual = compute_ydual(f, x, tx, contexts...)
    mypartials!(ty, ydual)
    return myvalue(ydual), ty
end

function DI.pushforward(
        f::F,
        prep::PisteOneArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ydual = compute_ydual(f, x, tx, contexts...)
    return mypartials(Val(B), ydual)
end

function DI.pushforward!(
        f::F,
        ty::NTuple,
        prep::PisteOneArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ydual = compute_ydual(f, x, tx, contexts...)
    return mypartials!(ty, ydual)
end

### Pushforward, two arguments

struct PisteTwoArgPushforwardPrep{SIG, YD} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    ydual::YD
end

function DI.prepare_pushforward_nokwarg(
        strict::Val,
        f!::F,
        y,
        backend::AutoPisteForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    ydual = similar(y, Dual{B, dual_eltype(x)})
    return PisteTwoArgPushforwardPrep(_sig, ydual)
end

function compute_ydual!(
        f!::F,
        prep::PisteTwoArgPushforwardPrep,
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    (; ydual) = prep
    T = dual_eltype(x)
    xdual = make_dual(T, x, tx)
    contexts_dual = translate(Dual{B, T}, contexts)
    f!(ydual, xdual, contexts_dual...)
    return ydual
end

function DI.value_and_pushforward(
        f!::F,
        y,
        prep::PisteTwoArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual = compute_ydual!(f!, prep, x, tx, contexts...)
    myvalue!(y, ydual)
    return y, mypartials(Val(B), ydual)
end

function DI.value_and_pushforward!(
        f!::F,
        y,
        ty::NTuple,
        prep::PisteTwoArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual = compute_ydual!(f!, prep, x, tx, contexts...)
    myvalue!(y, ydual)
    mypartials!(ty, ydual)
    return y, ty
end

function DI.pushforward(
        f!::F,
        y,
        prep::PisteTwoArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple{B},
        contexts::Vararg{DI.Context, C},
    ) where {F, B, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual = compute_ydual!(f!, prep, x, tx, contexts...)
    myvalue!(y, ydual)
    return mypartials(Val(B), ydual)
end

function DI.pushforward!(
        f!::F,
        y,
        ty::NTuple,
        prep::PisteTwoArgPushforwardPrep,
        backend::AutoPisteForward,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ydual = compute_ydual!(f!, prep, x, tx, contexts...)
    myvalue!(y, ydual)
    return mypartials!(ty, ydual)
end

### Gradient: Piste's own chunked driver, with a reusable workspace

# `Piste.gradient!` needs `x` and the gradient to be `Vector`s of one float type.
# Any other input falls back to DI's default (batched pushforwards, above).

struct PisteForwardGradientPrep{SIG, N, T} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    ws::GradientWorkspace{N, T}
end

# A fixed `chunksize` stays a compile-time constant, which is what lets this
# path compile under `juliac --trim`. With `nothing` the width is picked from
# `length(x)`, which is fine in the JIT but not statically known.
chunk(::AutoPisteForward{nothing}, K::Int) = pickchunk(K)
chunk(::AutoPisteForward{chunksize}, ::Int) where {chunksize} = Val(chunksize)

function DI.prepare_gradient_nokwarg(
        strict::Val,
        f::F,
        backend::AutoPisteForward,
        x::Vector{T},
        contexts::Vararg{DI.Context, C},
    ) where {F, T <: AbstractFloat, C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    ws = GradientWorkspace(x, chunk(backend, length(x)))
    return PisteForwardGradientPrep(_sig, ws)
end

function fix_contexts(
        f::F, ::GradientWorkspace{N, T}, contexts::Vararg{DI.Context, C}
    ) where {F, N, T, C}
    return DI.fix_tail(f, translate(Dual{N, T}, contexts)...)
end

function DI.value_and_gradient!(
        f::F,
        grad::Vector{T},
        prep::PisteForwardGradientPrep{SIG, N, T},
        backend::AutoPisteForward,
        x::Vector{T},
        contexts::Vararg{DI.Context, C},
    ) where {F, SIG, N, T, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = fix_contexts(f, prep.ws, contexts...)
    return value_and_gradient!(grad, fc, x, Val(N), prep.ws)
end

function DI.gradient!(
        f::F,
        grad::Vector{T},
        prep::PisteForwardGradientPrep{SIG, N, T},
        backend::AutoPisteForward,
        x::Vector{T},
        contexts::Vararg{DI.Context, C},
    ) where {F, SIG, N, T, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    fc = fix_contexts(f, prep.ws, contexts...)
    return gradient!(grad, fc, x, Val(N), prep.ws)
end

function DI.value_and_gradient!(
        f::F,
        grad,
        prep::PisteForwardGradientPrep,
        backend::AutoPisteForward,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
        f::F,
        grad,
        prep::PisteForwardGradientPrep,
        backend::AutoPisteForward,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

function DI.value_and_gradient(
        f::F,
        prep::PisteForwardGradientPrep,
        backend::AutoPisteForward,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return DI.value_and_gradient!(f, similar(x), prep, backend, x, contexts...)
end

function DI.gradient(
        f::F,
        prep::PisteForwardGradientPrep,
        backend::AutoPisteForward,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    return DI.gradient!(f, similar(x), prep, backend, x, contexts...)
end
