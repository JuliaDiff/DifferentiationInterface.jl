struct ReactantGradientPrep{SIG, XR, GR, CG, CG!, CVG, CVG!} <: DI.GradientPrep{SIG}
    _sig::Val{SIG}
    xr::XR
    gr::GR
    compiled_gradient::CG
    compiled_gradient!::CG!
    compiled_value_and_gradient::CVG
    compiled_value_and_gradient!::CVG!
end

function DI.prepare_gradient_nokwarg(
        strict::Val, f::F, rebackend::AutoReactant, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    _sig = DI.signature(f, rebackend, x; strict)
    backend = rebackend.mode
    xr = to_reac(x)
    gr = to_reac(similar(x))
    contextsr = map(to_reac, contexts)
    compiled_gradient = @compile DI.gradient(f, backend, xr, contextsr...)
    compiled_gradient! = @compile DI.gradient!(f, gr, backend, xr, contextsr...)
    compiled_value_and_gradient = @compile DI.value_and_gradient(f, backend, xr, contextsr...)
    compiled_value_and_gradient! = @compile DI.value_and_gradient!(f, gr, backend, xr, contextsr...)
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
        f::F, prep::ReactantGradientPrep, rebackend::AutoReactant, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, rebackend, x)
    backend = rebackend.mode
    (; xr, compiled_gradient) = prep
    copyto!(xr, x)
    contextsr = map(to_reac, contexts)
    gr = compiled_gradient(f, backend, xr, contextsr...)
    return gr
end

function DI.value_and_gradient(
        f::F, prep::ReactantGradientPrep, rebackend::AutoReactant, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, rebackend, x)
    backend = rebackend.mode
    (; xr, compiled_value_and_gradient) = prep
    copyto!(xr, x)
    contextsr = map(to_reac, contexts)
    yr, gr = compiled_value_and_gradient(f, backend, xr, contextsr...)
    return yr, gr
end

function DI.gradient!(
        f::F, grad, prep::ReactantGradientPrep, rebackend::AutoReactant, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, rebackend, x)
    backend = rebackend.mode
    (; xr, gr, compiled_gradient!) = prep
    copyto!(xr, x)
    contextsr = map(to_reac, contexts)
    compiled_gradient!(f, gr, backend, xr, contextsr...)
    return copyto!(grad, gr)
end

function DI.value_and_gradient!(
        f::F, grad, prep::ReactantGradientPrep, rebackend::AutoReactant, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, rebackend, x)
    backend = rebackend.mode
    (; xr, gr, compiled_value_and_gradient!) = prep
    copyto!(xr, x)
    contextsr = map(to_reac, contexts)
    yr, gr = compiled_value_and_gradient!(f, gr, backend, xr, contextsr...)
    return yr, copyto!(grad, gr)
end
