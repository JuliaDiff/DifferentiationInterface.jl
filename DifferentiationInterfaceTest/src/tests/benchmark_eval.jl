
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

    S1out = Scenario{op,:out,:out}
    S1in = Scenario{op,:in,:out}
    S2out = Scenario{op,:out,:in}
    S2in = Scenario{op,:in,:in}

    @eval function run_benchmark!(
        data::Vector{DifferentiationBenchmarkDataRow},
        backend::AbstractADType,
        scenario::Union{$S1out,$S1in,$S2out,$S2in};
        logging::Bool,
    )
        @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
            run_benchmark_aux(backend, scenario)
        catch exception
            logging && @warn "Error during benchmarking" backend scenario exception
            bench0, bench1, bench2 = failed_benchs(3)
            calls0, calls1, calls2 = -1, -1, -1
            (; bench0, bench1, bench2, calls0, calls1, calls2)
        end
        record!(data, backend, scenario, $prep_op, bench0, calls0)
        if scenario isa Union{$S1out,$S2out}
            record!(data, backend, scenario, $(string(val_and_op)), bench1, calls1)
            record!(data, backend, scenario, $(string(op)), bench2, calls2)
        elseif scenario isa Union{$S1in,$S2in}
            record!(data, backend, scenario, $(string(val_and_op!)), bench1, calls1)
            record!(data, backend, scenario, $(string(op!)), bench2, calls2)
        end
        return nothing
    end

    if op in [:derivative, :gradient, :jacobian]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be ex $val_and_op(f, _, ba, x, contexts...) evals = 1
            bench2 = @be ex $op(f, _, ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, ex, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, ex, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (res1, ex) $val_and_op!(f, _[1], _[2], ba, x, contexts...) evals =
                1
            bench2 = @be (res1, ex) $op!(f, _[1], _[2], ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, ex, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, res1, ex, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        op == :gradient && continue

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (y, ex) $val_and_op(f, _[1], _[2], ba, x, contexts...) evals = 1
            bench2 = @be (y, ex) $op(f, _[1], _[2], ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, y, ex, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, y, ex, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, res1, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (y, res1, ex) $val_and_op!(f, _[1], _[2], _[3], ba, x, contexts...) evals =
                1
            bench2 = @be (y, res1, ex) $op!(f, _[1], _[2], _[3], ba, x, contexts...) evals =
                1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, y, res1, ex, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, y, res1, ex, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:hessian, :second_derivative]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be ex $val_and_op(f, _, ba, x, contexts...) evals = 1
            bench2 = @be ex $op(f, _, ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, ex, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, ex, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1, res2, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (res1, res2, ex) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, contexts...
            ) evals = 1
            bench2 = @be (res2, ex) $op!(f, _[1], _[2], ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, res2, ex, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, res2, ex, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:pushforward, :pullback]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, tang, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be ex $val_and_op(f, _, ba, x, tang, contexts...) evals = 1
            bench2 = @be ex $op(f, _, ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, ex, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, ex, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, tang, res1, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be (res1, ex) $val_and_op!(f, _[1], _[2], ba, x, tang, contexts...) evals =
                1
            bench2 = @be (res1, ex) $op!(f, _[1], _[2], ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, ex, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, res1, ex, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y, tang, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be (y, ex) $val_and_op(f, _[1], _[2], ba, x, tang, contexts...) evals =
                1
            bench2 = @be (y, ex) $op(f, _[1], _[2], ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, y, ex, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, y, ex, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, tang, res1, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be (y, res1, ex) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, tang, contexts...
            ) evals = 1
            bench2 = @be (y, res1, ex) $op!(f, _[1], _[2], _[3], ba, x, tang, contexts...) evals =
                1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, y, res1, ex, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, y, res1, ex, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:hvp]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, tang, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be +(1, 1) evals = 1  # TODO: fix
            bench2 = @be ex $op(f, _, ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            calls1 = -1  # TODO: fix
            $op(cc, ex, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, tang, res2, contexts) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be +(1, 1) evals = 1  # TODO: fix
            bench2 = @be (res2, ex) $op!(f, _[1], _[2], ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            calls1 = -1  # TODO: fix
            $op!(cc, res2, ex, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end
    end
end
