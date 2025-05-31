@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:Union{Number,AbstractArray,Tuple}}

# nested vectors, similar are not supported
function Mooncake.rrule!!(
    dw::CoDual{<:DI.DifferentiateWith}, x::Union{CoDual{<:Number},CoDual{<:Tuple}}
)
    primal_func = primal(dw)
    primal_x = primal(x)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (y.dx,))
        @assert rdata(only(tx)) isa rdata_type(tangent_type(typeof(primal_x)))
        return NoRData(), rdata(only(tx))
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert rdata(only(tx)) isa rdata_type(tangent_type(typeof(primal_x)))
        return NoRData(), rdata(only(tx))
    end

    # output is a Tuple, NTuple
    function pullback_tuple!!(dy::Tuple)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert rdata(only(tx)) isa rdata_type(tangent_type(typeof(primal_x)))
        return NoRData(), rdata(only(tx))
    end

    # inputs are non Differentiable
    function pullback_nodiff!!(dy::NoRData)
        @assert tangent_type(typeof(primal(x))) <: NoTangent
        return NoRData(), dy
    end

    pullback = if tangent_type(typeof(primal(x))) <: NoTangent
        pullback_nodiff!!
    elseif typeof(primal(y)) <: Number
        pullback_scalar!!
    elseif typeof(primal(y)) <: Array
        pullback_array!!
    elseif typeof(primal(y)) <: Tuple
        pullback_tuple!!
    else
        error("$(typeof(primal(y))) primal type currently not supported.")
    end

    return y, pullback
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:AbstractArray})
    primal_func = primal(dw)
    primal_x = primal(x)
    fdata_arg = x.dx
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (y.dx,))
        @assert rdata(first(only(tx))) isa rdata_type(tangent_type(typeof(first(primal_x))))
        fdata_arg .+= only(tx)
        return NoRData(), dy
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert rdata(first(only(tx))) isa rdata_type(tangent_type(typeof(first(primal_x))))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    # output is a Tuple, NTuple
    function pullback_tuple!!(dy::Tuple)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert rdata(first(only(tx))) isa rdata_type(tangent_type(typeof(first(primal_x))))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    # inputs are non Differentiable
    function pullback_nodiff!!(dy::NoRData)
        @assert tangent_type(typeof(primal(x))) <: Vector{NoTangent}
        return NoRData(), dy
    end

    pullback = if tangent_type(typeof(primal(x))) <: Vector{NoTangent}
        pullback_nodiff!!
    elseif typeof(primal(y)) <: Number
        pullback_scalar!!
    elseif typeof(primal(y)) <: AbstractArray
        pullback_array!!
    elseif typeof(primal(y)) <: Tuple
        pullback_tuple!!
    else
        error("$(typeof(primal(y))) primal type currently not supported.")
    end

    return y, pullback
end

function Mooncake.generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:diffwith})
    return Any[], Any[]
end

function Mooncake.generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:diffwith})
    test_cases = reduce(
        vcat,
        map([(x) -> DI.DifferentiateWith(x, DI.AutoFiniteDiff())]) do F
            map([Float64, Float32]) do P
                return Any[
                    (false, :stability, nothing, F(cosh), P(0.3)),
                    (false, :stability, nothing, F(sinh), P(0.3)),
                    (false, :stability, nothing, F(Base.FastMath.exp10_fast), P(0.5)),
                    (false, :stability, nothing, F(Base.FastMath.exp2_fast), P(0.5)),
                    (false, :stability, nothing, F(Base.FastMath.exp_fast), P(5.0)),
                    (false, :none, nothing, F(copy), rand(Int32, 5)),
                ]
            end
        end...,
    )

    map([(x) -> DI.DifferentiateWith(x, DI.AutoZygote())]) do F
        map([Float64, Float32]) do P
            push!(
                test_cases,
                Any[
                    (false, :stability, nothing, F(Base.FastMath.sincos), P(3.0)),
                    (false, :none, nothing, F(Mooncake.__vec_to_tuple), Any[P(1.0)]),
                ]...,
            )
        end
    end

    map([(x) -> DI.DifferentiateWith(x, DI.AutoZygote())]) do F
        push!(
            test_cases,
            Any[
                (false, :stability, nothing, F(Mooncake.IntrinsicsWrappers.ctlz_int), 5),
                (false, :stability, nothing, F(Mooncake.IntrinsicsWrappers.ctpop_int), 5),
                (false, :stability, nothing, F(Mooncake.IntrinsicsWrappers.cttz_int), 5),
            ]...,
        )
    end

    map([(x) -> DI.DifferentiateWith(x, DI.AutoFiniteDiff())]) do F
        push!(
            test_cases,
            Any[
                (false, :stability, nothing, copy, randn(5, 4)),
                (
                    # Check that Core._apply_iterate gets lifted to _apply_iterate_equivalent.
                    false,
                    :none,
                    nothing,
                    F(x -> +(x...)),
                    randn(33),
                ),
                (
                    false,
                    :none,
                    nothing,
                    (F(
                        function (x)
                            rx = Ref(x)
                            return Base.pointerref(
                                Base.bitcast(Ptr{Float64}, pointer_from_objref(rx)), 1, 1
                            )
                        end,
                    )),
                    5.0,
                ),
                (false, :none, nothing, F(Mooncake.__vec_to_tuple), [1.0]),
                # (false, :none, nothing, F(Mooncake.__vec_to_tuple), Any[(1.0,)]), DI.basis fails for this, correct it!
                (false, :stability, nothing, F(Mooncake.IntrinsicsWrappers.ctlz_int), 5),
                (false, :stability, nothing, F(Mooncake.IntrinsicsWrappers.ctpop_int), 5),
                (false, :stability, nothing, F(Mooncake.IntrinsicsWrappers.cttz_int), 5),
                (
                    false,
                    :stability,
                    nothing,
                    F(Mooncake.IntrinsicsWrappers.abs_float),
                    5.0f0,
                ),
                (false, :stability, nothing, F(deepcopy), 5.0),
                (false, :stability, nothing, F(deepcopy), randn(5)),
                (false, :stability_and_allocs, nothing, F(sin), 1.1),
                (false, :stability_and_allocs, nothing, F(sin), 1.0f1),
                (false, :stability_and_allocs, nothing, F(cos), 1.1),
                (false, :stability_and_allocs, nothing, F(cos), 1.0f1),
                (false, :stability_and_allocs, nothing, F(exp), 1.1),
                (false, :stability_and_allocs, nothing, F(exp), 1.0f1),
            ]...,
        )
    end

    memory = Any[]
    return test_cases, memory
end
