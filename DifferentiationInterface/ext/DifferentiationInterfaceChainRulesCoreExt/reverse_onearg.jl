## Pullback

struct ChainRulesPullbackPrepSamePoint{Y,PB} <: DI.PullbackPrep
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f, ::AutoReverseChainRules, x, ty::NTuple, contexts::Vararg{DI.GeneralizedConstant,C}
) where {C}
    return DI.NoPullbackPrep()
end

function DI.prepare_pullback_same_point(
    f,
    ::DI.NoPullbackPrep,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x, map(DI.unwrap, contexts)...)
    return ChainRulesPullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(
    f,
    ::DI.NoPullbackPrep,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
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
    ::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    (; y, pb) = prep
    tx = map(ty) do dy
        unthunk(pb(dy)[2])
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::ChainRulesPullbackPrepSamePoint,
    ::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{DI.GeneralizedConstant,C},
) where {C}
    (; pb) = prep
    tx = map(ty) do dy
        unthunk(pb(dy)[2])
    end
    return tx
end
