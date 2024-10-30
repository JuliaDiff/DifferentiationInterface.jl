struct ReactantGradientPrep{XR,GR,CG,CG!,CVG,CVG!} <: GradientPrep
    xr::XR
    gr::GR
    compiled_gradient::CG
    compiled_gradient!::CG!
    compiled_value_and_gradient::CVG
    compiled_value_and_gradient!::CVG!
end

function DI.prepare_gradient(f, rebackend::ReactantBackend, x)
    (; backend) = rebackend
    xr = to_rarray(x)
    gr = to_rarray(similar(x))
    _gradient(_xr) = DI.gradient(f, backend, _xr)
    _gradient!(_gr, _xr) = DI.gradient!(f, _gr, backend, _xr)
    _value_and_gradient(_xr) = DI.value_and_gradient(f, backend, _xr)
    _value_and_gradient!(_gr, _xr) = DI.value_and_gradient!(f, _gr, backend, _xr)
    compiled_gradient = @compile _gradient(xr)
    compiled_gradient! = @compile _gradient!(gr, xr)
    compiled_value_and_gradient = @compile _value_and_gradient(xr)
    compiled_value_and_gradient! = @compile _value_and_gradient!(gr, xr)
    return ReactantGradientPrep(
        xr,
        gr,
        compiled_gradient,
        compiled_gradient!,
        compiled_value_and_gradient,
        compiled_value_and_gradient!,
    )
end

function DI.gradient(f, prep::ReactantGradientPrep, ::ReactantBackend, x)
    (; xr, compiled_gradient) = prep
    copyto!(xr, x)
    return compiled_gradient(xr)
end

function DI.value_and_gradient(f, prep::ReactantGradientPrep, ::ReactantBackend, x)
    (; xr, compiled_value_and_gradient) = prep
    copyto!(xr, x)
    return compiled_value_and_gradient(xr)
end

function DI.gradient!(f, grad, prep::ReactantGradientPrep, ::ReactantBackend, x)
    (; xr, gr, compiled_gradient!) = prep
    copyto!(xr, x)
    prep.compiled_gradient!(gr, xr)
    return copyto!(grad, gr)
end

function DI.value_and_gradient!(f, grad, prep::ReactantGradientPrep, ::ReactantBackend, x)
    (; xr, gr, compiled_value_and_gradient!) = prep
    copyto!(xr, x)
    y, gr = compiled_value_and_gradient!(gr, xr)
    return y, copyto!(grad, gr)
end
