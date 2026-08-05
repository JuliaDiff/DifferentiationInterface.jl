@eval function test_correctness(
        backend::AbstractADType,
        ts::TripleScenario;
        isapprox::Function,
        atol::Real,
        rtol::Real,
        reprepare::Bool,
    )
    ≈(x, y) = isapprox(x, y; atol, rtol)
    (; prep_scen, exec_scen, flush_scen) = deepcopy(ts)

    op = operator(ts)
    prep_op = preparator(op)
    prep_op! = preparator!(op)

    input_nonprepped_exec = clean_pre_backend_args(exec_scen)
    output_nonprepped_exec = op(
        func(exec_scen),
        input_nonprepped_exec...,
        backend,
        post_backend_args(exec_scen)...
    )
    result_nonprepped_exec = result_from_output(exec_scen, output_nonprepped_exec)

    prep = prep_op(
        func(prep_scen),
        clean_value_args(prep_scen)...,
        backend,
        post_backend_args(prep_scen)...
    )
    if reprepare && should_reprepare(ts)
        prep = prep_op!(
            func(flush_scen),
            clean_pre_backend_args(flush_scen)...,
            backend,
            post_backend_args(flush_scen)...
        )
    end

    input_prepped_exec1 = clean_pre_backend_args(exec_scen)
    output_prepped_exec1 = op(
        func(exec_scen),
        input_prepped_exec1...,
        prep,
        backend,
        post_backend_args(exec_scen)...
    )
    result_prepped_exec1 = result_from_output(exec_scen, output_prepped_exec1)

    op(
        func(flush_scen),
        clean_pre_backend_args(flush_scen)...,
        prep,
        backend,
        post_backend_args(flush_scen)...
    )

    input_prepped_exec2 = clean_pre_backend_args(exec_scen)
    output_prepped_exec2 = op(
        func(exec_scen),
        input_prepped_exec2...,
        prep,
        backend,
        post_backend_args(exec_scen)...
    )
    result_prepped_exec2 = result_from_output(exec_scen, output_prepped_exec2)

    # Test that the outputs are correct
    @testset "Correct values" begin
        @testset "Without prep" begin
            test_approx(
                op, result_nonprepped_exec, result(exec_scen);
                isapprox, atol, rtol
            )
        end
        @testset "First execution" begin
            test_approx(
                op, result_prepped_exec1, result(exec_scen);
                isapprox, atol, rtol
            )
        end
        @testset "Second execution" begin
            test_approx(
                op, result_prepped_exec2, result(exec_scen);
                isapprox, atol, rtol
            )
        end
    end

    # Test that the inputs are returned
    @testset "Buffer forwarding" begin
        @testset "Without prep" begin
            test_same(
                op, result_nonprepped_exec, result_from_input(exec_scen, input_nonprepped_exec)
            )
        end
        @testset "First execution" begin
            test_same(
                op, result_prepped_exec1, result_from_input(exec_scen, input_prepped_exec1)
            )
        end
    end

    # Test that the scenario is intact
    @testset "Scenario intact" begin
        @test protected_post_backend_args(exec_scen) == protected_post_backend_args(ts.exec_scen)
    end

    return nothing
end
