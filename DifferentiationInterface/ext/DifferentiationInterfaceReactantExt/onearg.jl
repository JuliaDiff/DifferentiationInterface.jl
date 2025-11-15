struct ReactantGradientPrep{SIG, XR, GR, CG, CG!, CVG, CVG!} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    xr::XR
    gr::GR
    compiled_gradient::CG
    compiled_gradient!::CG!
    compiled_value_and_gradient::CVG
    compiled_value_and_gradient!::CVG!
end

function DI.prepare_gradient_nokwarg(strict::Val, f::F, rebackend::AutoReactant, x) where {F}
    _sig = DI.signature(f, rebackend, x; strict)
    backend = rebackend.mode
    xr = to_rarray(x)
    gr = to_rarray(similar(x))
    _gradient(_xr) = DI.gradient(f, backend, _xr)
    _gradient!(_gr, _xr) = copy!(_gr, DI.gradient(f, backend, _xr))
    _value_and_gradient(_xr) = DI.value_and_gradient(f, backend, _xr)
    function _value_and_gradient!(_gr, _xr)
        y, __gr = DI.value_and_gradient(f, backend, _xr)
        copy!(_gr, __gr)
        return y, _gr
    end
    compiled_gradient = @compile _gradient(xr)
    compiled_gradient! = @compile _gradient!(gr, xr)
    compiled_value_and_gradient = @compile _value_and_gradient(xr)
    compiled_value_and_gradient! = @compile _value_and_gradient!(gr, xr)
    return ReactantGradientPrep(
        _sig,
        xr,
        gr,
        compiled_gradient,
        compiled_gradient!,
        compiled_value_and_gradient,
        compiled_value_and_gradient!,
    )
end

function DI.gradient(
        f::F, prep::ReactantGradientPrep, rebackend::AutoReactant, x
    ) where {F}
    DI.check_prep(f, prep, rebackend, x)
    (; xr, compiled_gradient) = prep
    copy!(xr, x)
    gr = compiled_gradient(xr)
    g = convert(typeof(x), gr)
    return g
end

function DI.value_and_gradient(
        f::F, prep::ReactantGradientPrep, rebackend::AutoReactant, x
    ) where {F}
    DI.check_prep(f, prep, rebackend, x)
    (; xr, compiled_value_and_gradient) = prep
    copy!(xr, x)
    yr, gr = compiled_value_and_gradient(xr)
    y = convert(eltype(x), yr)
    g = convert(typeof(x), gr)
    return y, g
end

function DI.gradient!(
        f::F, grad, prep::ReactantGradientPrep, rebackend::AutoReactant, x
    ) where {F}
    DI.check_prep(f, prep, rebackend, x)
    (; xr, gr, compiled_gradient!) = prep
    copy!(xr, x)
    compiled_gradient!(gr, xr)
    return copy!(grad, gr)
end

function DI.value_and_gradient!(
        f::F, grad, prep::ReactantGradientPrep, rebackend::AutoReactant, x
    ) where {F}
    DI.check_prep(f, prep, rebackend, x)
    (; xr, gr, compiled_value_and_gradient!) = prep
    copy!(xr, x)
    yr, gr = compiled_value_and_gradient!(gr, xr)
    y = convert(eltype(x), yr)
    return y, copy!(grad, gr)
end
