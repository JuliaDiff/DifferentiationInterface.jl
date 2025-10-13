function test_noallocs(skip::Bool, func, args...)
    possible_allocations = check_allocs(func, typeof.(args); ignore_throw = true)
    return @test isempty(possible_allocations) skip = skip
end

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
        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1out; subset::Symbol, skip::Bool
            )
            (; f, x, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                test_noallocs(skip, $prep_op, f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) && test_noallocs(skip, $op, f, ba, x, contexts...)
            (subset == :full) && test_noallocs(skip, $val_and_op, f, ba, x, contexts...)
            (subset != :none) && test_noallocs(skip, $op, f, prep, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op, f, prep, ba, x, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1in; subset::Symbol, skip::Bool
            )
            (; f, x, res1, contexts, prep_args) = deepcopy(scen)
            res1_sim = mysimilar(res1)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                test_noallocs(skip, $prep_op, f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                test_noallocs(skip, $op!, f, res1_sim, prep, ba, x, contexts...)
            (subset == :full) &&
                test_noallocs(skip, $val_and_op!, f, res1_sim, prep, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $op!, f, res1_sim, prep, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op!, f, res1_sim, prep, ba, x, contexts...)
            return nothing
        end

        op == :gradient && continue

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S2out; subset::Symbol, skip::Bool
            )
            (; f, x, y, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) && test_noallocs(
                skip, $prep_op, f, prep_args.y, ba, prep_args.x, prep_args.contexts...
            )
            (subset == :full) && test_noallocs(skip, $op, f, y, ba, x, contexts...)
            (subset == :full) && test_noallocs(skip, $val_and_op, f, y, ba, x, contexts...)
            (subset != :none) && test_noallocs(skip, $op, f, y, prep, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op, f, y, prep, ba, x, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S2in; subset::Symbol, skip::Bool
            )
            (; f, x, y, res1, contexts, prep_args) = deepcopy(scen)
            res1_sim = mysimilar(res1)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) && test_noallocs(
                skip, $prep_op, f, prep_args.y, ba, prep_args.x, prep_args.contexts...
            )
            (subset == :full) &&
                test_noallocs(skip, $op!, f, y, res1_sim, ba, x, contexts...)
            (subset == :full) &&
                test_noallocs(skip, $val_and_op!, f, y, res1_sim, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $op!, f, y, res1_sim, prep, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op!, f, y, res1_sim, prep, ba, x, contexts...)
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1out; subset::Symbol, skip::Bool
            )
            (; f, x, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                test_noallocs(skip, $prep_op, f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) && test_noallocs(skip, $op, f, ba, x, contexts...)
            (subset == :full) && test_noallocs(skip, $val_and_op, f, ba, x, contexts...)
            (subset != :none) && test_noallocs(skip, $op, f, prep, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op, f, prep, ba, x, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1in; subset::Symbol, skip::Bool
            )
            (; f, x, res1, res2, contexts, prep_args) = deepcopy(scen)
            res1_sim, res2_sim = mysimilar(res1), mysimilar(res2)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) &&
                test_noallocs(skip, $prep_op, f, ba, prep_args.x, prep_args.contexts...)
            (subset == :full) && test_noallocs(skip, $op!, f, res2_sim, ba, x, contexts...)
            (subset == :full) &&
                test_noallocs(skip, $val_and_op!, f, res1_sim, res2_sim, ba, x, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $op!, f, res2_sim, prep, ba, x, contexts...)
            (subset != :none) && test_noallocs(
                skip, $val_and_op!, f, res1_sim, res2_sim, prep, ba, x, contexts...
            )
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1out; subset::Symbol, skip::Bool
            )
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) && test_noallocs(
                skip, $prep_op, f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) && test_noallocs(skip, $op, f, ba, x, t, contexts...)
            (subset == :full) && test_noallocs(skip, $val_and_op, f, ba, x, t, contexts...)
            (subset != :none) && test_noallocs(skip, $op, f, prep, ba, x, t, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op, f, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1in; subset::Symbol, skip::Bool
            )
            (; f, x, t, res1, contexts, prep_args) = deepcopy(scen)
            res1_sim = mysimilar(res1)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) && test_noallocs(
                skip, $prep_op, f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                test_noallocs(skip, $op!, f, res1_sim, ba, x, t, contexts...)
            (subset == :full) &&
                test_noallocs(skip, $val_and_op!, f, res1_sim, ba, x, t, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $op!, f, res1_sim, prep, ba, x, t, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op!, f, res1_sim, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S2out; subset::Symbol, skip::Bool
            )
            (; f, x, y, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) && test_noallocs(
                skip,
                $prep_op,
                f,
                prep_args.y,
                ba,
                prep_args.x,
                prep_args.t,
                prep_args.contexts...,
            )
            (subset == :full) && test_noallocs(skip, $op, f, y, ba, x, t, contexts...)
            (subset == :full) &&
                test_noallocs(skip, $val_and_op, f, y, ba, x, t, contexts...)
            (subset != :none) && test_noallocs(skip, $op, f, y, prep, ba, x, t, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op, f, y, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S2in; subset::Symbol, skip::Bool
            )
            (; f, x, y, t, res1, contexts, prep_args) = deepcopy(scen)
            res1_sim = mysimilar(res1)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) && test_noallocs(
                skip,
                $prep_op,
                f,
                prep_args.y,
                ba,
                prep_args.x,
                prep_args.t,
                prep_args.contexts...,
            )
            (subset == :full) &&
                test_noallocs(skip, $op!, f, y, res1_sim, ba, x, t, contexts...)
            (subset == :full) &&
                test_noallocs(skip, $val_and_op!, f, y, res1_sim, ba, x, t, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $op!, f, y, res1_sim, prep, ba, x, t, contexts...)
            (subset != :none) && test_noallocs(
                skip, $val_and_op!, f, y, res1_sim, prep, ba, x, t, contexts...
            )
            return nothing
        end

    elseif op in [:hvp]
        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1out; subset::Symbol, skip::Bool
            )
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) && test_noallocs(
                skip, $prep_op, f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) && test_noallocs(skip, $val_and_op, f, ba, x, t, contexts...)
            (subset == :full) && test_noallocs(skip, $op, f, ba, x, t, contexts...)
            (subset != :none) &&
                test_noallocs(skip, $val_and_op, f, prep, ba, x, t, contexts...)
            (subset != :none) && test_noallocs(skip, $op, f, prep, ba, x, t, contexts...)
            return nothing
        end

        @eval function test_alloccheck(
                ba::AbstractADType, scen::$S1in; subset::Symbol, skip::Bool
            )
            (; f, x, t, res1, res2, contexts, prep_args) = deepcopy(scen)
            res1_sim, res2_sim = mysimilar(res1), mysimilar(res2)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            (subset == :full) && test_noallocs(
                skip, $prep_op, f, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            (subset == :full) &&
                test_noallocs(skip, $op!, f, res2_sim, ba, x, t, contexts...)
            (subset == :full) && test_noallocs(
                skip, $val_and_op!, f, res1_sim, res2_sim, ba, x, t, contexts...
            )
            (subset != :none) &&
                test_noallocs(skip, $op!, f, res2_sim, prep, ba, x, t, contexts...)
            (subset != :none) && test_noallocs(
                skip, $val_and_op!, f, res1_sim, res2_sim, prep, ba, x, t, contexts...
            )
            return nothing
        end
    end
end
