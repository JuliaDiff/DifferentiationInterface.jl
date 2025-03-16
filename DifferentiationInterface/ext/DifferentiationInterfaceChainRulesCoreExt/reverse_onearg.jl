## Pullback

struct ChainRulesPullbackPrepSamePoint{SIG,Y,PB} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C};
    strict::Val=Val(false),
) where {C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    return DI.NoPullbackPrep(_sig)
end

function DI.prepare_pullback_same_point(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C};
    strict::Val=Val(false),
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x, map(DI.unwrap, contexts)...)
    return ChainRulesPullbackPrepSamePoint(_sig, y, pb)
end

function DI.value_and_pullback(
    f,
    prep::DI.NoPullbackPrep,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x, map(DI.unwrap, contexts)...)
    tx = map(ty) do dy
        unthunk(pb(dy)[2])
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::ChainRulesPullbackPrepSamePoint,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; y, pb) = prep
    tx = map(ty) do dy
        unthunk(pb(dy)[2])
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::ChainRulesPullbackPrepSamePoint,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    (; pb) = prep
    tx = map(ty) do dy
        unthunk(pb(dy)[2])
    end
    return tx
end
