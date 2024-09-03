for op in [
    :derivative,
    :gradient,
    :hessian,
    :hvp,
    :jacobian,
    :pullback,
    :pushforward,
    :second_derivative,
]
    op! = Symbol(op, "!")
    val_prefix = if op == :second_derivative
        "value_derivative_and_"
    elseif op in [:hessian, :hvp]
        "value_gradient_and_"
    else
        "value_and_"
    end
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)

    S1out = Scenario{op,1,:outofplace}
    S1in = Scenario{op,1,:inplace}
    S2out = Scenario{op,2,:outofplace}
    S2in = Scenario{op,2,:inplace}

    if op in [:derivative, :gradient, :jacobian]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op(f, ex, ba, x)
            JET.@test_call $op(f, ex, ba, x)
            JET.@test_opt $val_and_op(f, ex, ba, x)
            JET.@test_call $val_and_op(f, ex, ba, x)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op!(f, res1, ex, ba, x)
            JET.@test_call $op!(f, res1, ex, ba, x)
            JET.@test_opt $val_and_op!(f, res1, ex, ba, x)
            JET.@test_call $val_and_op!(f, res1, ex, ba, x)
        end

        op == :gradient && continue

        @eval function test_jet(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x)
            JET.@test_opt $op(f, y, ex, ba, x)
            JET.@test_call $op(f, y, ex, ba, x)
            JET.@test_opt $val_and_op(f, y, ex, ba, x)
            JET.@test_call $val_and_op(f, y, ex, ba, x)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, res1) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x)
            JET.@test_opt $op!(f, y, res1, ex, ba, x)
            JET.@test_call $op!(f, y, res1, ex, ba, x)
            JET.@test_opt $val_and_op!(f, y, res1, ex, ba, x)
            JET.@test_call $val_and_op!(f, y, res1, ex, ba, x)
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op(f, ex, ba, x)
            JET.@test_call $op(f, ex, ba, x)
            JET.@test_opt $val_and_op(f, ex, ba, x)
            JET.@test_call $val_and_op(f, ex, ba, x)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1, res2) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op!(f, res2, ex, ba, x)
            JET.@test_call $op!(f, res2, ex, ba, x)
            JET.@test_opt $val_and_op!(f, res1, res2, ex, ba, x)
            JET.@test_call $val_and_op!(f, res1, res2, ex, ba, x)
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, seed) = deepcopy(scen)
            ex = $prep_op(f, ba, x, seed)
            JET.@test_opt $op(f, ex, ba, x, seed)
            JET.@test_call $op(f, ex, ba, x, seed)
            JET.@test_opt $val_and_op(f, ex, ba, x, seed)
            JET.@test_call $val_and_op(f, ex, ba, x, seed)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, seed, res1, res2) = deepcopy(scen)
            ex = $prep_op(f, ba, x, seed)
            JET.@test_opt $op!(f, res1, ex, ba, x, seed)
            JET.@test_call $op!(f, res1, ex, ba, x, seed)
            JET.@test_opt $val_and_op!(f, res1, ex, ba, x, seed)
            JET.@test_call $val_and_op!(f, res1, ex, ba, x, seed)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y, seed) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x, seed)
            JET.@test_opt $op(f, y, ex, ba, x, seed)
            JET.@test_call $op(f, y, ex, ba, x, seed)
            JET.@test_opt $val_and_op(f, y, ex, ba, x, seed)
            JET.@test_call $val_and_op(f, y, ex, ba, x, seed)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, seed, res1) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x, seed)
            JET.@test_opt $op!(f, y, res1, ex, ba, x, seed)
            JET.@test_call $op!(f, y, res1, ex, ba, x, seed)
            JET.@test_opt $val_and_op!(f, y, res1, ex, ba, x, seed)
            JET.@test_call $val_and_op!(f, y, res1, ex, ba, x, seed)
        end

    elseif op in [:hvp]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, seed) = deepcopy(scen)
            ex = $prep_op(f, ba, x, seed)
            JET.@test_opt $op(f, ex, ba, x, seed)
            JET.@test_call $op(f, ex, ba, x, seed)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, seed, res1, res2) = deepcopy(scen)
            ex = $prep_op(f, ba, x, seed)
            JET.@test_opt $op!(f, res2, ex, ba, x, seed)
            JET.@test_call $op!(f, res2, ex, ba, x, seed)
        end
    end
end
