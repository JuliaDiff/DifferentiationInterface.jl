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

    @eval function DIT.run_benchmark!(
            data::DifferentiationBenchmark,
            backend::AbstractADType,
            scenario::Union{$S1out, $S1in, $S2out, $S2in};
            logging::Bool,
            subset::Symbol,
            count_calls::Bool,
            benchmark_test::Bool,
            benchmark_seconds::Real,
            benchmark_aggregation,
        )
        @assert subset in (:full, :prepared)

        bench_success = true
        bench_result = try
            benchmark_aux(backend, scenario; subset, s = benchmark_seconds)
        catch exception
            bench_success = false
            logging && @warn "Error during benchmarking" backend scenario exception
            BenchmarkResult()
        end
        benchmark_test && @test bench_success

        if count_calls
            count_success = true
            calls_result = try
                calls_aux(backend, scenario; subset, s = nothing)
            catch exception
                count_success = false
                logging && @warn "Error during call counting" backend scenario exception
                CallsResult()
            end
            benchmark_test && @test count_success
        else
            calls_result = CallsResult()
        end

        prep_string = $(string(prep_op))
        if scenario isa Union{$S1out, $S2out}
            valop_string = $(string(val_and_op))
            op_string = $(string(op))
        else
            valop_string = $(string(val_and_op!))
            op_string = $(string(op!))
        end

        record!(
            data;
            backend,
            scenario,
            operator = valop_string,
            prepared = true,
            bench = bench_result.prepared_valop,
            calls = calls_result.prepared_valop,
            aggregation = benchmark_aggregation,
        )
        record!(
            data;
            backend,
            scenario,
            operator = op_string,
            prepared = true,
            bench = bench_result.prepared_op,
            calls = calls_result.prepared_op,
            aggregation = benchmark_aggregation,
        )
        if subset == :full
            record!(
                data;
                backend,
                scenario,
                operator = prep_string,
                prepared = nothing,
                bench = bench_result.preparation,
                calls = calls_result.preparation,
                aggregation = benchmark_aggregation,
            )
            record!(
                data;
                backend,
                scenario,
                operator = valop_string,
                prepared = false,
                bench = bench_result.unprepared_valop,
                calls = calls_result.unprepared_valop,
                aggregation = benchmark_aggregation,
            )
            record!(
                data;
                backend,
                scenario,
                operator = op_string,
                prepared = false,
                bench = bench_result.unprepared_op,
                calls = calls_result.unprepared_op,
                aggregation = benchmark_aggregation,
            )
        end
        return nothing
    end

    if op in [:derivative, :gradient, :jacobian]
        @eval function benchmark_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            prepared_valop = @be prep $val_and_op(f, _, ba, x, contexts...) seconds = s
            prepared_op = @be prep $op(f, _, ba, x, contexts...) seconds = s
            if subset == :full
                preparation = @be $prep_op(f, ba, prep_args.x, prep_args.contexts...) seconds =
                    s
                unprepared_valop = @be $val_and_op(f, ba, x, contexts...) seconds = s
                unprepared_op = @be $op(f, ba, x, contexts...) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, contexts...)
            prepared_valop = reset_count!(cc)
            $op(cc, prep, ba, x, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op(cc, ba, x, contexts...)
            unprepared_valop = reset_count!(cc)
            $op(cc, ba, x, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            prepared_valop = @be (mysimilar(res1), prep) $val_and_op!(
                f, _[1], _[2], ba, x, contexts...
            ) seconds = s
            prepared_op = @be (mysimilar(res1), prep) $op!(
                f, _[1], _[2], ba, x, contexts...
            ) seconds = s
            if subset == :full
                preparation = @be $prep_op(f, ba, prep_args.x, prep_args.contexts...) seconds =
                    s
                unprepared_valop = @be mysimilar(res1) $val_and_op!(
                    f, _, ba, x, contexts...
                ) seconds = s
                unprepared_op = @be mysimilar(res1) $op!(f, _, ba, x, contexts...) seconds =
                    s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, res1, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), prep, ba, x, contexts...)
            prepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res1), prep, ba, x, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), ba, x, contexts...)
            unprepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res1), ba, x, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        op == :gradient && continue

        @eval function benchmark_aux(ba::AbstractADType, scen::$S2out; subset::Symbol, s)
            (; f, x, y, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            prepared_valop = @be (y, prep) $val_and_op(f, _[1], _[2], ba, x, contexts...) seconds =
                s
            prepared_op = @be (y, prep) $op(f, _[1], _[2], ba, x, contexts...) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, prep_args.y, ba, prep_args.x, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be y $val_and_op(f, _, ba, x, contexts...) seconds = s
                unprepared_op = @be y $op(f, _, ba, x, contexts...) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S2out; subset::Symbol, s)
            (; f, x, y, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op(cc, y, prep, ba, x, contexts...)
            prepared_valop = reset_count!(cc)
            $op(cc, y, prep, ba, x, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op(cc, y, ba, x, contexts...)
            unprepared_valop = reset_count!(cc)
            $op(cc, y, ba, x, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S2in; subset::Symbol, s)
            (; f, x, y, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            prepared_valop = @be (y, mysimilar(res1), prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, contexts...
            ) seconds = s
            prepared_op = @be (y, mysimilar(res1), prep) $op!(
                f, _[1], _[2], _[3], ba, x, contexts...
            ) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, prep_args.y, ba, prep_args.x, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be (y, mysimilar(res1)) $val_and_op!(
                    f, _[1], _[2], ba, x, contexts...
                ) seconds = s
                unprepared_op = @be (y, mysimilar(res1)) $op!(
                    f, _[1], _[2], ba, x, contexts...
                ) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S2in; subset::Symbol, s)
            (; f, x, y, res1, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, prep_args.y, ba, prep_args.x, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op!(cc, y, mysimilar(res1), prep, ba, x, contexts...)
            prepared_valop = reset_count!(cc)
            $op!(cc, y, mysimilar(res1), prep, ba, x, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op!(cc, y, mysimilar(res1), ba, x, contexts...)
            unprepared_valop = reset_count!(cc)
            $op!(cc, y, mysimilar(res1), ba, x, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

    elseif op in [:hessian, :second_derivative]
        @eval function benchmark_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            prepared_valop = @be prep $val_and_op(f, _, ba, x, contexts...) seconds = s
            prepared_op = @be prep $op(f, _, ba, x, contexts...) seconds = s
            if subset == :full
                preparation = @be $prep_op(f, ba, prep_args.x, prep_args.contexts...) seconds =
                    s
                unprepared_valop = @be $val_and_op(f, ba, x, contexts...) seconds = s
                unprepared_op = @be $op(f, ba, x, contexts...) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, contexts...)
            prepared_valop = reset_count!(cc)
            $op(cc, prep, ba, x, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op(cc, ba, x, contexts...)
            unprepared_valop = reset_count!(cc)
            $op(cc, ba, x, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, res1, res2, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.contexts...)
            prepared_valop = @be (mysimilar(res1), mysimilar(res2), prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, contexts...
            ) seconds = s
            prepared_op = @be (mysimilar(res2), prep) $op!(
                f, _[1], _[2], ba, x, contexts...
            ) seconds = s
            if subset == :full
                preparation = @be $prep_op(f, ba, prep_args.x, prep_args.contexts...) seconds =
                    s
                unprepared_valop = @be (mysimilar(res1), mysimilar(res2)) $val_and_op!(
                    f, _[1], _[2], ba, x, contexts...
                ) seconds = s
                unprepared_op = @be mysimilar(res2) $op!(f, _, ba, x, contexts...) seconds =
                    s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, res1, res2, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), mysimilar(res2), prep, ba, x, contexts...)
            prepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res2), prep, ba, x, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), mysimilar(res2), ba, x, contexts...)
            unprepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res2), ba, x, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

    elseif op in [:pushforward, :pullback]
        @eval function benchmark_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            prepared_valop = @be prep $val_and_op(f, _, ba, x, t, contexts...) seconds = s
            prepared_op = @be prep $op(f, _, ba, x, t, contexts...) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, ba, prep_args.x, prep_args.t, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be $val_and_op(f, ba, x, t, contexts...) seconds = s
                unprepared_op = @be $op(f, ba, x, t, contexts...) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end
        @eval function calls_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, t, contexts...)
            prepared_valop = reset_count!(cc)
            $op(cc, prep, ba, x, t, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op(cc, ba, x, t, contexts...)
            unprepared_valop = reset_count!(cc)
            $op(cc, ba, x, t, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, t, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            prepared_valop = @be (mysimilar(res1), prep) $val_and_op!(
                f, _[1], _[2], ba, x, t, contexts...
            ) seconds = s
            prepared_op = @be (mysimilar(res1), prep) $op!(
                f, _[1], _[2], ba, x, t, contexts...
            ) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, ba, prep_args.x, prep_args.t, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be mysimilar(res1) $val_and_op!(
                    f, _, ba, x, t, contexts...
                ) seconds = s
                unprepared_op = @be mysimilar(res1) $op!(f, _, ba, x, t, contexts...) seconds =
                    s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, t, res1, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), prep, ba, x, t, contexts...)
            prepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res1), prep, ba, x, t, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), ba, x, t, contexts...)
            unprepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res1), ba, x, t, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S2out; subset::Symbol, s)
            (; f, x, y, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            prepared_valop = @be (y, prep) $val_and_op(f, _[1], _[2], ba, x, t, contexts...)
            prepared_op = @be (y, prep) $op(f, _[1], _[2], ba, x, t, contexts...)
            if subset == :full
                preparation = @be $prep_op(
                    f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be y $val_and_op(f, _, ba, x, t, contexts...) seconds =
                    s
                unprepared_op = @be y $op(f, _, ba, x, t, contexts...) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S2out; subset::Symbol, s)
            (; f, x, y, t, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(
                cc, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            preparation = reset_count!(cc)
            $val_and_op(cc, y, prep, ba, x, t, contexts...)
            prepared_valop = reset_count!(cc)
            $op(cc, y, prep, ba, x, t, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op(cc, y, ba, x, t, contexts...)
            unprepared_valop = reset_count!(cc)
            $op(cc, y, ba, x, t, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S2in; subset::Symbol, s)
            (; f, x, y, t, res1, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(
                f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            prepared_valop = @be (y, mysimilar(res1), prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, t, contexts...
            ) seconds = s
            prepared_op = @be (y, mysimilar(res1), prep) $op!(
                f, _[1], _[2], _[3], ba, x, t, contexts...
            ) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be (y, mysimilar(res1)) $val_and_op!(
                    f, _[1], _[2], ba, x, t, contexts...
                ) seconds = s
                unprepared_op = @be (y, mysimilar(res1)) $op!(
                    f, _[1], _[2], ba, x, t, contexts...
                ) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S2in; subset::Symbol, s)
            (; f, x, y, t, res1, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(
                cc, prep_args.y, ba, prep_args.x, prep_args.t, prep_args.contexts...
            )
            preparation = reset_count!(cc)
            $val_and_op!(cc, y, mysimilar(res1), prep, ba, x, t, contexts...)
            prepared_valop = reset_count!(cc)
            $op!(cc, y, mysimilar(res1), prep, ba, x, t, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op!(cc, y, mysimilar(res1), ba, x, t, contexts...)
            unprepared_valop = reset_count!(cc)
            $op!(cc, y, mysimilar(res1), ba, x, t, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

    elseif op in [:hvp]
        @eval function benchmark_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            prepared_valop = @be prep $val_and_op(f, _, ba, x, t, contexts...) seconds = s
            prepared_op = @be prep $op(f, _, ba, x, t, contexts...) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, ba, prep_args.x, prep_args.t, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be $val_and_op(f, ba, x, t, contexts...) seconds = s
                unprepared_op = @be $op(f, ba, x, t, contexts...) seconds = s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1out; subset::Symbol, s)
            (; f, x, t, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, t, contexts...)
            prepared_valop = reset_count!(cc)
            $op(cc, prep, ba, x, t, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op(cc, ba, x, t, contexts...)
            unprepared_valop = reset_count!(cc)
            $op(cc, ba, x, t, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end

        @eval function benchmark_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, t, res1, res2, contexts, prep_args) = deepcopy(scen)
            prep = $prep_op(f, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            prepared_valop = @be (mysimilar(res1), mysimilar(res2), prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, t, contexts...
            ) seconds = s
            prepared_op = @be (mysimilar(res2), prep) $op!(
                f, _[1], _[2], ba, x, t, contexts...
            ) seconds = s
            if subset == :full
                preparation = @be $prep_op(
                    f, ba, prep_args.x, prep_args.t, prep_args.contexts...
                ) seconds = s
                unprepared_valop = @be (mysimilar(res1), mysimilar(res2)) $val_and_op!(
                    f, _[1], _[2], ba, x, t, contexts...
                ) seconds = s
                unprepared_op = @be mysimilar(res2) $op!(f, _, ba, x, t, contexts...) seconds =
                    s
                return BenchmarkResult(;
                    prepared_valop,
                    prepared_op,
                    preparation,
                    unprepared_valop,
                    unprepared_op,
                )
            else
                return BenchmarkResult(; prepared_valop, prepared_op)
            end
        end

        @eval function calls_aux(ba::AbstractADType, scen::$S1in; subset::Symbol, s)
            (; f, x, t, res1, res2, contexts, prep_args) = deepcopy(scen)
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, prep_args.x, prep_args.t, prep_args.contexts...)
            preparation = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), mysimilar(res2), prep, ba, x, t, contexts...)
            prepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res2), prep, ba, x, t, contexts...)
            prepared_op = reset_count!(cc)
            $val_and_op!(cc, mysimilar(res1), mysimilar(res2), ba, x, t, contexts...)
            unprepared_valop = reset_count!(cc)
            $op!(cc, mysimilar(res2), ba, x, t, contexts...)
            unprepared_op = reset_count!(cc)
            return CallsResult(;
                prepared_valop, prepared_op, preparation, unprepared_valop, unprepared_op
            )
        end
    end
end
