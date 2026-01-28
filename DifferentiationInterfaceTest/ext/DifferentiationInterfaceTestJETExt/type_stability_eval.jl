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

    S1out = Scenario{op, :out, :out}
    S1in = Scenario{op, :in, :out}
    S2out = Scenario{op, :out, :in}
    S2in = Scenario{op, :in, :in}

    if op in [:derivative, :gradient, :jacobian]
        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1out;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, prep, ba, x, contexts...)
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1in;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res1), ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(f, mysimilar(res1), ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res1), prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, mysimilar(res1), prep, ba, x, contexts...
            )
            return nothing
        end

        op == :gradient && continue

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S2out;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, y, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, y, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, y, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, y, prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, y, prep, ba, x, contexts...)
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S2in;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, y, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, y, mysimilar(res1), ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(f, y, mysimilar(res1), ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, y, mysimilar(res1), prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, y, mysimilar(res1), prep, ba, x, contexts...
            )
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1out;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, prep, ba, x, contexts...)
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1in;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, res1, res2, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res2), ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, mysimilar(res1), mysimilar(res2), ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res2), prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, mysimilar(res1), mysimilar(res2), prep, ba, x, contexts...
            )
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1out;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, ba, x, t, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, prep, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1in;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, t, res1, res2, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res1), ba, x, t, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(f, mysimilar(res1), ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res1), prep, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, mysimilar(res1), prep, ba, x, t, contexts...
            )
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S2out;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, y, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, y, ba, x, t, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, y, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, y, prep, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, y, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S2in;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, y, t, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, y, mysimilar(res1), ba, x, t, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, y, mysimilar(res1), ba, x, t, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, y, mysimilar(res1), prep, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, y, mysimilar(res1), prep, ba, x, t, contexts...
            )
            return nothing
        end

    elseif op in [:hvp]
        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1out;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, ba, x, t, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op(f, prep, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op(f, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function DIT.test_jet(
                ba::AbstractADType,
                scen::$S1in;
                subset::Symbol,
                ignored_modules,
                function_filter,
            )
            (; f, x, t, res1, res2, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $prep_op(
                f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res2), ba, x, t, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, mysimilar(res1), mysimilar(res2), ba, x, t, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $op!(f, mysimilar(res2), prep, ba, x, t, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules function_filter =
                function_filter $val_and_op!(
                f, mysimilar(res1), mysimilar(res2), prep, ba, x, t, contexts...
            )
            return nothing
        end
    end
end
