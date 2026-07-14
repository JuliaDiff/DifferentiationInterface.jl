struct MooncakeHVPPrep{SIG, Tcache} <: DI.HVPPrep{SIG}
    _sig::Val{SIG}
    cache::Tcache
end

function DI.prepare_hvp_nokwarg(
        strict::Val, f::F, backend::AutoMooncakeForwardOverReverse, x, tx::NTuple, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    config = get_config(backend)
    cache = prepare_hvp_cache(f, x, map(DI.unwrap, contexts)...; config)
    prep = MooncakeHVPPrep(_sig, cache)
    return prep
end

function DI.gradient_and_hvp(
        f::F,
        prep::MooncakeHVPPrep{Y},
        backend::AutoMooncakeForwardOverReverse,
        x,
        tx::NTuple{1},
        contexts::Vararg{DI.Context, C},
    ) where {F, Y, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    dx = only(tx)
    _, (_, new_g), (_, new_dg) = value_and_hvp!!(
        prep.cache, f, dx, x, map(DI.unwrap, contexts)...
    )
    return _copy_output(new_g), _copy_output(new_dg)
end

function DI.gradient_and_hvp(
        f::F,
        prep::MooncakeHVPPrep{Y},
        backend::AutoMooncakeForwardOverReverse,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, Y, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    gs_and_tg = map(tx) do dx
        _, (_, new_g), (_, new_dg) = value_and_hvp!!(
            prep.cache, f, dx, x, map(DI.unwrap, contexts)...
        )
        _copy_output(new_g), _copy_output(new_dg)
    end
    g = first(gs_and_tg[1])
    tg = map(last, gs_and_tg)
    return g, tg
end

function DI.gradient_and_hvp!(
        f::F,
        tg::NTuple,
        prep::MooncakeHVPPrep,
        backend::AutoMooncakeForwardOverReverse,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    y, new_tg = DI.gradient_and_hvp(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, tg, new_tg)
    return y, tx
end

function DI.hvp(
        f::F,
        prep::MooncakeHVPPrep,
        backend::AutoMooncakeForwardOverReverse,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.gradient_and_hvp(f, prep, backend, x, tx, contexts...)[2]
end

function DI.hvp!(
        f::F,
        tg::NTuple,
        prep::MooncakeHVPPrep,
        backend::AutoMooncakeForwardOverReverse,
        x,
        tx::NTuple,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.gradient_and_hvp!(f, tg, prep, backend, x, tx, contexts...)[2]
end
