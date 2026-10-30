using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using DifferentiationInterface:
    AutoSimpleFiniteDiff,
    AutoForwardFromPrimitive,
    AutoReverseFromPrimitive,
    DenseSparsityDetector
using SparseMatrixColorings
using JLArrays, StaticArrays
using JET: JET
using Test

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
        @test jacobian(copyto!, zeros(10), backend, ones(10)) isa AbstractMatrix
    end
end

@testset "Misc" begin
    @test_throws ArgumentError DifferentiationInterface.overloaded_input(
        pushforward, sum, AutoSimpleFiniteDiff(), 1, (1, 2)
    )
    @test_throws ArgumentError DifferentiationInterface.overloaded_input(
        pushforward, copyto!, [1.0], AutoSimpleFiniteDiff(), [1.0], ([1.0], [1.0])
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
    # multiple batches
    forward_backend = AutoSimpleFiniteDiff(; chunksize = 2)
    reverse_backend = AutoReverseFromPrimitive(AutoSimpleFiniteDiff(; chunksize = 2))
    multibatch_scenarios = filter(
        s -> DIT.operator(s) in (:jacobian, :hessian), static_scenarios()
    )
    test_differentiation(
        [forward_backend, reverse_backend], multibatch_scenarios; logging = LOGGING
    )

    # JET < 0.12 does not define `JET_AVAILABLE`
    if !isdefined(JET, :JET_AVAILABLE) || JET.JET_AVAILABLE
        inplace_hessian(s) = DIT.operator(s) == :hessian && DIT.operator_place(s) == :in
        test_differentiation(
            forward_backend,
            multibatch_scenarios;
            correctness = false,
            type_stability = :prepared,
            logging = LOGGING,
        )
        test_differentiation(
            reverse_backend,
            filter(!inplace_hessian, multibatch_scenarios);
            correctness = false,
            type_stability = :prepared,
            logging = LOGGING,
        )
        @testset "Type stability (broken)" for scen in filter(inplace_hessian, multibatch_scenarios)
            # the inner pullback-based gradient fails to infer in `hessian!`
            (; f, x, res1, res2, prep_args) = deepcopy(scen)
            prep = prepare_hessian(f, reverse_backend, prep_args.x)
            JET.@test_opt broken = true hessian!(
                f, DIT.mysimilar(res2), prep, reverse_backend, x
            )
            JET.@test_opt broken = true value_gradient_and_hessian!(
                f, DIT.mysimilar(res1), DIT.mysimilar(res2), prep, reverse_backend, x
            )
        end
    end
end;

@testset "Array format preservation in wrong mode" begin
    @test gradient(sum, AutoSimpleFiniteDiff(), jl(ones(2))) isa JLVector
    @test derivative(t -> jl(fill(t, 2)), AutoReverseFromPrimitive(AutoSimpleFiniteDiff()), 1.0) isa JLVector
end
