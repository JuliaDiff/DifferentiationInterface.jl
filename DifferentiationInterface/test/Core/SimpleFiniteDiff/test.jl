using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface:
    AutoSimpleFiniteDiff,
    AutoForwardFromPrimitive,
    AutoReverseFromPrimitive,
    DenseSparsityDetector
using SparseMatrixColorings
using JLArrays, StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

backends = [ #
    AutoSimpleFiniteDiff(; chunksize = 5),
    AutoForwardFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 4)),
    AutoReverseFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 4)),
]

second_order_backends = [ #
    SecondOrder(
        AutoForwardFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 5)),
        AutoReverseFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 4)),
    ),
    SecondOrder(
        AutoReverseFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 5)),
        AutoForwardFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 4)),
    ),
]

second_order_hvp_backends = [ #
    SecondOrder(
        AutoReverseFromPrimitive(AutoSimpleFiniteDiff(); inplace = false),
        AutoForwardFromPrimitive(AutoSimpleFiniteDiff()),
    ),
    SecondOrder(
        AutoForwardFromPrimitive(AutoSimpleFiniteDiff(); inplace = false),
        AutoReverseFromPrimitive(AutoSimpleFiniteDiff()),
    ),
    SecondOrder(
        AutoForwardFromPrimitive(AutoSimpleFiniteDiff(); inplace = false),
        AutoForwardFromPrimitive(AutoSimpleFiniteDiff()),
    ),
    SecondOrder(
        AutoReverseFromPrimitive(AutoSimpleFiniteDiff(); inplace = false),
        AutoReverseFromPrimitive(AutoSimpleFiniteDiff()),
    ),
]

adaptive_backends = [ #
    AutoSimpleFiniteDiff(),
    AutoReverseFromPrimitive(AutoSimpleFiniteDiff()),
    SecondOrder(AutoSimpleFiniteDiff(), AutoReverseFromPrimitive(AutoSimpleFiniteDiff())),
    SecondOrder(AutoReverseFromPrimitive(AutoSimpleFiniteDiff()), AutoSimpleFiniteDiff()),
]

for backend in vcat(backends, second_order_backends)
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense scenarios

@testset "Dense" begin
    test_differentiation(
        vcat(backends, second_order_backends),
        default_scenarios(; include_constantified = true, include_smaller = true);
        logging = LOGGING,
    )

    test_differentiation(
        second_order_hvp_backends,
        default_scenarios(; include_constantorcachified = true);
        excluded = vcat(FIRST_ORDER, :hessian, :second_derivative),
        logging = LOGGING,
    )

    test_differentiation(
        vcat(
            backends[2:3],
            AutoReverseFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 1))
        ),
        complex_scenarios();
        logging = LOGGING
    )
end

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse.(adaptive_backends),
        default_scenarios(; include_constantified = true);
        logging = LOGGING,
    )

    test_differentiation(
        MyAutoSparse.(
            vcat(adaptive_backends, MixedMode(adaptive_backends[1], adaptive_backends[2]))
        ),
        sparse_scenarios(;
            include_constantified = true,
            include_cachified = true,
            include_constantorcachified = true,
            use_tuples = true,
        );
        sparsity = true,
        logging = LOGGING,
    )

    @testset "Complex numbers" begin
        test_differentiation(
            AutoSparse.(
                vcat(
                    adaptive_backends, MixedMode(adaptive_backends[1], adaptive_backends[2])
                );
                sparsity_detector = DenseSparsityDetector(AutoSimpleFiniteDiff(); atol = 1.0e-5),
                coloring_algorithm = GreedyColoringAlgorithm(),
            ),
            complex_sparse_scenarios();
            logging = LOGGING,
        )
    end

    @testset "SparseMatrixColorings access" begin
        jac_for_prep = prepare_jacobian(copy, MyAutoSparse(adaptive_backends[1]), rand(10))
        jac_rev_prep = prepare_jacobian(copy, MyAutoSparse(adaptive_backends[2]), rand(10))
        hess_prep = prepare_hessian(
            x -> sum(abs2, x), MyAutoSparse(adaptive_backends[1]), rand(10)
        )

        @test all(==(1), column_colors(jac_for_prep))
        @test all(==(1), row_colors(jac_rev_prep))
        @test all(==(1), column_colors(hess_prep))
        @test ncolors(jac_for_prep) == 1
        @test ncolors(hess_prep) == 1
        @test only(column_groups(jac_for_prep)) == 1:10
        @test only(row_groups(jac_rev_prep)) == 1:10
        @test only(column_groups(hess_prep)) == 1:10
    end

    @testset "Empty color groups in sparse AD" begin # issue 857
        # forward
        backend = MyAutoSparse(adaptive_backends[1])
        @test jacobian(zero, backend, ones(10)) isa AbstractMatrix
        @test hessian(sum ∘ zero, backend, ones(10)) isa AbstractMatrix
        # reverse
        backend = MyAutoSparse(adaptive_backends[2])
        @test jacobian(zero, backend, ones(10)) isa AbstractMatrix
        @test hessian(sum ∘ zero, backend, ones(10)) isa AbstractMatrix
        # mixed
        backend = MyAutoSparse(MixedMode(adaptive_backends[1], adaptive_backends[2]))
        @test jacobian(copy!, zeros(10), backend, ones(10)) isa AbstractMatrix
    end
end

@testset "Misc" begin
    @test_throws ArgumentError DifferentiationInterface.overloaded_input(
        pushforward, sum, AutoSimpleFiniteDiff(), 1, (1, 2)
    )
    @test_throws ArgumentError DifferentiationInterface.overloaded_input(
        pushforward, copy!, [1.0], AutoSimpleFiniteDiff(), [1.0], ([1.0], [1.0])
    )
end

@testset "Weird arrays" begin
    test_differentiation(
        [
            AutoSimpleFiniteDiff(),
            AutoForwardFromPrimitive(AutoSimpleFiniteDiff()),
            AutoReverseFromPrimitive(AutoSimpleFiniteDiff()),
        ],
        vcat(static_scenarios(), gpu_scenarios());
        logging = LOGGING,
    )
end;
