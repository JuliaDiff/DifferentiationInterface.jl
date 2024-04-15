## Pullback

DI.prepare_pullback(f!, y, ::AutoReverseEnzyme, x) = NoPullbackExtras()

function DI.value_and_pullback(
    f!, y, ::AutoReverseEnzyme, x::Number, dy, ::NoPullbackExtras
)
    dy_sametype = convert(typeof(y), copy(dy))
    _, new_dx = only(autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Active(x)))
    return y, new_dx
end

function DI.value_and_pullback(
    f!, y, ::AutoReverseEnzyme, x::AbstractArray, dy, ::NoPullbackExtras
)
    dx_sametype = zero(x)
    dy_sametype = convert(typeof(y), copy(dy))
    autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype))
    return y, dx_sametype
end
