include("../../testutils.jl")

using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: DenseSparsityDetector
using FiniteDiff: FiniteDiff
using SparseMatrixColorings
using Test

using ExplicitImports
check_no_implicit_imports(DifferentiationInterface)

for backend in [AutoFiniteDiff()]
    @test check_available(backend)
    @test check_inplace(backend)
    @test DifferentiationInterface.inner_preparation_behavior(backend) isa
        DifferentiationInterface.PrepareInnerSimple
end

@testset "Dense" begin
    test_differentiation(
        AutoFiniteDiff(),
        default_scenarios(;
            include_constantified = true,
            include_cachified = true,
            include_constantorcachified = true,
            use_tuples = true,
            include_smaller = true,
        );
        excluded = [:second_derivative, :hvp],
        logging = LOGGING,
    )

    test_differentiation(
        SecondOrder(AutoFiniteDiff(; relstep = 1.0e-5, absstep = 1.0e-5), AutoFiniteDiff()),
        default_scenarios();
        logging = LOGGING,
        rtol = 1.0e-2,
    )

    test_differentiation(
        [
            AutoFiniteDiff(; relstep = cbrt(eps(Float64))),
            AutoFiniteDiff(; relstep = cbrt(eps(Float64)), absstep = cbrt(eps(Float64))),
            AutoFiniteDiff(; dir = 0.5),
        ];
        excluded = [:second_derivative, :hvp],
        logging = LOGGING,
    )
end

@testset "Sparse" begin
    test_differentiation(
        MyAutoSparse(AutoFiniteDiff()),
        sparse_scenarios();
        excluded = SECOND_ORDER,
        logging = LOGGING,
    )
end

@testset "Complex" begin
    test_differentiation(AutoFiniteDiff(), complex_scenarios(); logging = LOGGING)
    test_differentiation(
        AutoSparse(
            AutoFiniteDiff();
            sparsity_detector = DenseSparsityDetector(AutoFiniteDiff(); atol = 1.0e-5),
            coloring_algorithm = GreedyColoringAlgorithm(),
        ),
        complex_sparse_scenarios();
        logging = LOGGING,
    )
end;

@testset "Step size" begin  # fix 811
    backend = AutoFiniteDiff(; absstep = 1000, relstep = 0.1)
    preps = [
        prepare_pushforward(identity, backend, 1.0, (1.0,)),
        prepare_pushforward(copyto!, [0.0], backend, [1.0], ([1.0],)),
        prepare_derivative(identity, backend, 1.0),
        prepare_derivative((y, x) -> y .= x, [0.0], backend, 1.0),
        prepare_gradient(sum, backend, [1.0]),
        prepare_jacobian(identity, backend, [1.0]),
        prepare_jacobian(copyto!, [0.0], backend, [1.0]),
    ]
    for prep in preps
        @test prep.relstep == 0.1
        @test prep.absstep == 1000
    end
    prep = prepare_hessian(sum, backend, [1.0])
    @test prep.absstep_g == 1000
    @test prep.absstep_h == 1000
    @test prep.relstep_g == 0.1
    @test prep.relstep_h == 0.1

    backend = AutoFiniteDiff(; relstep = 0.1)
    preps = [
        prepare_pushforward(identity, backend, 1.0, (1.0,)),
        prepare_pushforward(copyto!, [0.0], backend, [1.0], ([1.0],)),
        prepare_derivative(identity, backend, 1.0),
        prepare_derivative((y, x) -> y .= x, [0.0], backend, 1.0),
        prepare_gradient(sum, backend, [1.0]),
        prepare_jacobian(identity, backend, [1.0]),
        prepare_jacobian(copyto!, [0.0], backend, [1.0]),
    ]
    for prep in preps
        @test prep.relstep == 0.1
        @test prep.absstep == 0.1
    end
    prep = prepare_hessian(sum, backend, [1.0])
    @test prep.absstep_g == 0.1
    @test prep.absstep_h == 0.1
    @test prep.relstep_g == 0.1
    @test prep.relstep_h == 0.1
end

include("benchmark.jl")
