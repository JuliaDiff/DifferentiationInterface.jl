## Pushforward

function DI.value_and_pushforward!!(
    f!, y, dy, backend::AutoForwardEnzyme, x, dx, extras::Nothing
)
    dx_sametype = convert(typeof(x), copy(dx))
    dy_sametype = convert(typeof(y), dy)
    autodiff(
        backend.mode, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype)
    )
    return y, myupdate!!(dy, dy_sametype)
end
