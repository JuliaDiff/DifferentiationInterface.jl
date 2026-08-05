function DIT.test_type_stability(
        backend::AbstractADType,
        ts::TripleScenario;
        subset::TestSubset,
        ignored_modules,
        function_filter,
    )
    (; prep_scen, exec_scen) = ts
    op = operator(ts)
    prep_op = preparator(op)

    prep = prep_op(
        pre_backend_args(prep_scen)...,
        backend,
        post_backend_args(prep_scen)...
    )

    # prepared operator
    @test_opt ignored_modules = ignored_modules function_filter =
        function_filter op(
        func(exec_scen),
        pre_backend_args(exec_scen)...,
        prep,
        backend,
        post_backend_args(exec_scen)...
    )

    if subset == TEST_FULL
        # unprepared operator
        @test_opt ignored_modules = ignored_modules function_filter =
            function_filter op(
            func(exec_scen),
            pre_backend_args(exec_scen)...,
            backend,
            post_backend_args(exec_scen)...
        )

        # preparation itself
        @test_opt ignored_modules = ignored_modules function_filter =
            function_filter prep_op(
            func(prep_scen),
            pre_backend_args(prep_scen)...,
            backend,
            post_backend_args(exec_scen)...
        )
    end

    return nothing
end
