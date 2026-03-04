function ChainRulesCore.rrule(
        dw::DI.DifferentiateWith{C}, x, contexts::Vararg{Any, C}
    ) where {C}
    (; f, backend, context_wrappers) = dw
    y = f(x, contexts...)
    wrapped_contexts = map(DI.call, context_wrappers, contexts)
    prep_same = DI.prepare_pullback_same_point_nokwarg(
        Val(false), f, backend, x, (y,), wrapped_contexts...
    )
    function diffwith_pullbackfunc(dy)
        dx = DI.pullback(f, prep_same, backend, x, (dy,), wrapped_contexts...) |> only
        dc = map(contexts) do c
            @not_implemented(
                """
                Derivatives with respect to context arguments are not implemented.
                """
            )
        end
        return (NoTangent(), dx, dc...)
    end
    return y, diffwith_pullbackfunc
end
