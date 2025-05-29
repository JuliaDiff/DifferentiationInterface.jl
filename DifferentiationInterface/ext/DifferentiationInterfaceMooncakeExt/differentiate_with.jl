@is_primitive MinimalCtx Tuple{DI.DifferentiateWith,<:Union{Number,AbstractArray,Tuple}}

function Mooncake.rrule!!(
    dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{Union{<:Number,<:Tuple}}
)
    primal_func = primal(dw)
    primal_x = primal(x)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (fdata(y.dx),))
        @assert only(tx) isa rdata_type(typeof(primal_x))
        return NoRData(), only(tx)
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert only(tx) isa rdata_type(typeof(primal_x))
        return NoRData(), only(tx)
    end

    # output is a Tuple, NTuple
    function pullback_tuple!!(dy::Tuple)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert only(tx) isa rdata_type(typeof(primal_x))
        return NoRData(), only(tx)
    end

    pullback = if typeof(primal(y)) <: Number
        pullback_scalar!!
    elseif typeof(primal(y)) <: Array
        pullback_array!!
    else
        pullback_tuple!!
    end

    return y, pullback
end

function Mooncake.rrule!!(dw::CoDual{<:DI.DifferentiateWith}, x::CoDual{<:AbstractArray})
    primal_func = primal(dw)
    primal_x = primal(x)
    fdata_arg = fdata(x.dx)
    (; f, backend) = primal_func
    y = zero_fcodual(f(primal_x))

    # output is a vector, so we need to use the vector pullback
    function pullback_array!!(dy::NoRData)
        tx = DI.pullback(f, backend, primal_x, (fdata(y.dx),))
        @assert first(only(tx)) isa rdata_type(typeof(first(primal_x)))
        fdata_arg .+= only(tx)
        return NoRData(), dy
    end

    # output is a scalar, so we can use the scalar pullback
    function pullback_scalar!!(dy::Number)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert first(only(tx)) isa rdata_type(typeof(first(primal_x)))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    # output is a Tuple, NTuple
    function pullback_tuple!!(dy::Tuple)
        tx = DI.pullback(f, backend, primal_x, (dy,))
        @assert first(only(tx)) isa rdata_type(typeof(first(primal_x)))
        fdata_arg .+= only(tx)
        return NoRData(), NoRData()
    end

    pullback = if typeof(primal(y)) <: Number
        pullback_scalar!!
    elseif typeof(primal(y)) <: Array
        pullback_array!!
    else
        pullback_tuple!!
    end

    return y, pullback
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:diffwith})
    return Any[], Any[]
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:diffwith})
    test_cases = reduce(
        vcat,
        map([Float64, Float32]) do P
            return Any[
                (false, :stability_and_allocs, nothing, cosh, P(0.3)),
                (false, :stability_and_allocs, nothing, sinh, P(0.3)),
                (false, :stability_and_allocs, nothing, Base.FastMath.exp10_fast, P(0.5)),
                (false, :stability_and_allocs, nothing, Base.FastMath.exp2_fast, P(0.5)),
                (false, :stability_and_allocs, nothing, Base.FastMath.exp_fast, P(5.0)),
                (false, :stability_and_allocs, nothing, Base.FastMath.sincos, P(3.0)),
            ]
        end,
    )
    push!(test_cases, (false, :stability, nothing, copy, randn(5, 4)))
    push!(test_cases, (
        # Check that Core._apply_iterate gets lifted to _apply_iterate_equivalent.
        false,
        :none,
        nothing,
        x -> +(x...),
        randn(33),
    ))
    push!(
        test_cases,
        (
            false,
            :none,
            nothing,
            (
                function (x)
                    rx = Ref(x)
                    return Base.pointerref(
                        Base.bitcast(Ptr{Float64}, pointer_from_objref(rx)), 1, 1
                    )
                end
            ),
            5.0,
        ),
    )
    push!(
        test_cases,
        (
            false,
            :none,
            nothing,
            x -> (pointerset(pointer(x), UInt8(3), 2, 1); x),
            rand(UInt8, 5),
        ),
    )
    push!(test_cases, (false, :none, nothing, Mooncake.__vec_to_tuple, [1.0]))
    push!(test_cases, (false, :none, nothing, Mooncake.__vec_to_tuple, Any[1.0]))
    push!(test_cases, (false, :none, nothing, Mooncake.__vec_to_tuple, Any[[1.0]]))
    push!(test_cases, (false, :stability, nothing, Mooncake.IntrinsicsWrappers.ctlz_int, 5))
    push!(
        test_cases, (false, :stability, nothing, Mooncake.IntrinsicsWrappers.ctpop_int, 5)
    )
    push!(test_cases, (false, :stability, nothing, Mooncake.IntrinsicsWrappers.cttz_int, 5))
    push!(
        test_cases, (false, :stability, nothing, Mooncake.IntrinsicsWrappers.abs_float, 5.0)
    )
    push!(
        test_cases,
        (false, :stability, nothing, Mooncake.IntrinsicsWrappers.abs_float, 5.0f0),
    )
    push!(test_cases, (false, :stability, nothing, deepcopy, 5.0))
    push!(test_cases, (false, :stability, nothing, deepcopy, randn(5)))
    push!(test_cases, (false, :stability_and_allocs, nothing, sin, 1.1))
    push!(test_cases, (false, :stability_and_allocs, nothing, sin, 1.0f1))
    push!(test_cases, (false, :stability_and_allocs, nothing, cos, 1.1))
    push!(test_cases, (false, :stability_and_allocs, nothing, cos, 1.0f1))
    push!(test_cases, (false, :stability_and_allocs, nothing, exp, 1.1))
    push!(test_cases, (false, :stability_and_allocs, nothing, exp, 1.0f1))

    # additional_test_set = Mooncake.tangent_test_cases()
    # function is_valid(f)
    #     try
    #         isa(f([1.0, 2.0]), Union{<:Number,<:AbstractArray})
    #     catch
    #         false
    #     end
    # end
    # for test in additional_test_set
    #     if is_valid(test[2])
    #         push!(test_cases, test)
    #     end
    # end

    memory = Any[]
    return test_cases, memory
end
