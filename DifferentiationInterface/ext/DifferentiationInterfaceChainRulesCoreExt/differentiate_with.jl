function ChainRulesCore.rrule(dw::DI.DifferentiateWith, x)
    (; f, backend) = dw
    y = f(x)
    prep_same = DI.prepare_pullback_same_point_nokwarg(Val(true), f, backend, x, (y,))
    function pullbackfunc(dy)
        tx = DI.pullback(f, prep_same, backend, x, (dy,))
        return (NoTangent(), only(tx))
    end
    return y, pullbackfunc
end
