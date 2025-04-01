function define_rule!(primal_func, primal_args)
    return eval(:(@from_rrule MinimalCtx Tuple{$primal_func,$primal_args...}))
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, args::CoDual...)
    primal_func = typeof(Mooncake.primal(dw))
    primal_args = typeof.(map(arg -> Mooncake.primal(arg), args))
    # use the DI.chainrule wrapper inside @from_rrule to create a custom rrule!!

    # macro evaluation in global scope with more specialized types (@fromrrule requires non generic types)
    define_rule!(primal_func, primal_args)

    # Use the ChainRuleCore rrule mapping with backends, calling Mooncake rule!! that now wraps around that ChainRulesCore rrule.
    return Base.invokelatest(Mooncake.rrule!!, CoDual(primal_func, dw.dx), args...)
end
