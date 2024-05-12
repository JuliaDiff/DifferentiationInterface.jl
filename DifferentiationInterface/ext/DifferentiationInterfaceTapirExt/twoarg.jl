struct TapirTwoArgPullbackExtras{R} <: PullbackExtras
    rrule::R
end

function DI.prepare_pullback(f!, y, backend::AutoTapir, x, dy)
    rrule = build_rrule(f!, y, x)
    extras = TapirTwoArgPullbackExtras(rrule)
    DI.value_and_pullback(f!, y, backend, x, dy, extras)  # warm up
    return extras
end

# see https://github.com/withbayes/Tapir.jl/issues/113#issuecomment-2036718992

function DI.value_and_pullback(f!, y, ::AutoTapir, x, dy, extras::TapirTwoArgPullbackExtras)
    dy_righttype = convert(tangent_type(typeof(y)), copy(dy))
    dx_righttype = zero_tangent(x)

    # We want the VJP, not VJP + dx, so I'm going to zero-out `dx`. `set_to_zero!!` has the advantage
    # that it will also replace any immutable components of `dx` to zero.
    dx_righttype = set_to_zero!!(dx_righttype)

    # We want `dy` to correspond to the cotangent of `y` _after_
    # running the forwards-pass, so I'm going to take a copy, and zero-out the original.
    dy_righttype_backup = copy(dy_righttype)
    dy_righttype = set_to_zero!!(dy_righttype)

    # Mutate a copy of `y`, so that we can run the reverse-pass later on.
    y_copy = copy(y)

    # In case `f!` is a closure
    df! = zero_tangent(f!)

    # Run the forwards-pass.
    out, pb!! = extras.rrule(
        CoDual(f!, fdata(df!)),
        CoDual(y_copy, fdata(dy_righttype)),
        CoDual(x, fdata(dx_righttype)),
    )

    # Verify that the output is non-differentiable.
    @assert primal(out) === nothing

    # Set the cotangent of `y` to be equal to the requested value.
    dy_righttype = increment!!(dy_righttype, dy_righttype_backup)

    # Record the state of `y` before running the reverse-pass.
    y = copyto!(y, y_copy)

    # Run the reverse-pass.
    _, _, new_dx = pb!!(NoRData())

    return y, tangent(fdata(dx_righttype), new_dx)
end
