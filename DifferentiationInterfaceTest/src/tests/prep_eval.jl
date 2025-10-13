const PME = PreparationMismatchError

for op in ALL_OPS
    op! = Symbol(op, "!")
    val_prefix = if op == :second_derivative
        "value_derivative_and_"
    elseif op == :hessian
        "value_gradient_and_"
    elseif op == :hvp
        "gradient_and_"
    else
        "value_and_"
    end
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)
    prep_op! = Symbol("prepare!_", op)
    prep_op_same = Symbol("prepare_", op, "_same_point")

    P = if op == :derivative
        DerivativePrep
    elseif op == :gradient
        GradientPrep
    elseif op == :hessian
        HessianPrep
    elseif op == :hvp
        HVPPrep
    elseif op == :jacobian
        JacobianPrep
    elseif op == :pullback
        PullbackPrep
    elseif op == :pushforward
        PushforwardPrep
    elseif op == :second_derivative
        SecondDerivativePrep
    end

    S1out = Scenario{op, :out, :out}
    S1in = Scenario{op, :in, :out}
    S2out = Scenario{op, :out, :in}
    S2in = Scenario{op, :in, :in}

    if op in [:derivative, :gradient, :jacobian]
        @eval function test_prep(ba::AbstractADType, scen::$S1out)
            (; f, x, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op(nothing, prep, ba, x, contexts...)
            @test_throws PME $op(nothing, prep, ba, x, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S1in)
            (; f, x, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), prep, ba, x, contexts...
            )
            @test_throws PME $op!(nothing, mysimilar(res1), prep, ba, x, contexts...)
            return nothing
        end

        op == :gradient && continue

        @eval function test_prep(ba::AbstractADType, scen::$S2out)
            (; f, x, y, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op(nothing, mysimilar(y), prep, ba, x, contexts...)
            @test_throws PME $op(nothing, mysimilar(y), prep, ba, x, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S2in)
            (; f, x, y, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op!(
                nothing, mysimilar(y), mysimilar(res1), prep, ba, x, contexts...
            )
            @test_throws PME $op!(
                nothing, mysimilar(y), mysimilar(res1), prep, ba, x, contexts...
            )
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_prep(ba::AbstractADType, scen::$S1out)
            (; f, x, y, res1, res2, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op(nothing, prep, ba, x, contexts...)
            @test_throws PME $op(nothing, prep, ba, x, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S1in)
            (; f, x, y, res1, res2, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), mysimilar(res2), prep, ba, x, contexts...
            )
            @test_throws PME $op!(nothing, mysimilar(res2), prep, ba, x, contexts...)
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_prep(ba::AbstractADType, scen::$S1out)
            (; f, x, y, t, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op(nothing, prep, ba, x, t, contexts...)
            @test_throws PME $op(nothing, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S1in)
            (; f, x, y, t, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), prep, ba, x, t, contexts...
            )
            @test_throws PME $op!(nothing, mysimilar(res1), prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S2out)
            (; f, x, y, t, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            @test prep isa $P
            @test_throws PME $val_and_op(nothing, mysimilar(y), prep, ba, x, t, contexts...)
            @test_throws PME $op(nothing, mysimilar(y), prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S2in)
            (; f, x, y, t, res1, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            @test prep isa $P
            @test_throws PME $val_and_op!(
                nothing, mysimilar(y), mysimilar(res1), prep, ba, x, t, contexts...
            )
            @test_throws PME $op!(
                nothing, mysimilar(y), mysimilar(res1), prep, ba, x, t, contexts...
            )
            return nothing
        end

    elseif op in [:hvp]
        @eval function test_prep(ba::AbstractADType, scen::$S1out)
            (; f, x, y, t, res1, res2, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $val_and_op(nothing, prep, ba, x, t, contexts...)
            @test_throws PME $op(nothing, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_prep(ba::AbstractADType, scen::$S1in)
            (; f, x, y, t, res1, res2, contexts, prep_args) = new_scen = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            @test prep isa $P
            @test_throws PME $op!(nothing, mysimilar(res2), prep, ba, x, t, contexts...)
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), mysimilar(res2), prep, ba, x, t, contexts...
            )
            return nothing
        end
    end
end
