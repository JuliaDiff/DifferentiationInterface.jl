## Pushforward

function DI.prepare_pushforward(f!, y, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, dx)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    ::NoPushforwardExtras,
)
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = convert(typeof(x), dx)
    dy_sametype = make_zero(y)
    y_and_dy = Duplicated(y, dy_sametype)
    x_and_dx = Duplicated(x, dx_sametype)
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(forward_mode(backend), f!_and_df!, Const, y_and_dy, x_and_dx)
    else
        autodiff(forward_mode(backend), f!_and_df!, Const, y_and_dy, x_and_dx)
    end
    return y, dy_sametype
end
