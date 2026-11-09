using ADTypes
using DifferentiationInterface
using DifferentiationInterface:
    AutoSimpleFiniteDiff,
    BatchSizeSettings,
    pick_batchsize,
    reasonable_batchsize,
    threshold_batchsize
import DifferentiationInterface as DI
using LinearAlgebra: I
using StaticArrays
using Test

BSS = BatchSizeSettings

@testset "Default" begin
    @test (@inferred pick_batchsize(AutoZygote(), zeros(0))) isa BSS{1, false, true}
    @test (@inferred pick_batchsize(AutoZygote(), zeros(2))) isa BSS{1, false, true}
    @test (@inferred pick_batchsize(AutoZygote(), zeros(100))) isa BSS{1, false, true}
    @test_throws ArgumentError pick_batchsize(AutoSparse(AutoZygote()), zeros(2))
    @test_throws ArgumentError pick_batchsize(
        SecondOrder(AutoZygote(), AutoZygote()), zeros(2)
    )
    @test_throws ArgumentError pick_batchsize(
        MixedMode(AutoSimpleFiniteDiff(), AutoZygote()), zeros(2)
    )
end

@testset "Constructor" begin
    @test BSS(1, 0) === BSS{1, false, true}(0, 0)
    @test BSS(0, 0) === BSS{0, true, true}(0, 0)
    @test BSS(1, 1) === BSS{1, true, true}(1, 1)
    @test BSS(2, 2) === BSS{2, true, true}(2, 1)
    @test BSS(4, 8) === BSS{4, false, true}(8, 2)
    @test BSS(8, 13) === BSS{8, false, false}(13, 2)
    @test_throws ArgumentError BSS(3, 2)
end

@testset "SimpleFiniteDiff (adaptive)" begin
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(0))) isa BSS{1, false, true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(2))) isa BSS{2, true, true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(6))) isa BSS{6, true, true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(12))) isa BSS{12, true, true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(24))) isa BSS{12, false, true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(100))) isa BSS{12, false, false}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(0)))) isa
        BSS{0, true, true}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(2)))) isa
        BSS{2, true, true}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(6)))) isa
        BSS{6, true, true}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(100)))) isa
        BSS{100, true, true}
end

@testset "SimpleFiniteDiff (fixed)" begin
    @test_throws ArgumentError pick_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), zeros(2))
    @test_throws ArgumentError pick_batchsize(
        AutoSimpleFiniteDiff(; chunksize = 4), @SVector(zeros(2))
    )
    @test pick_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), zeros(6)) isa BSS{4}
    @test pick_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), zeros(100)) isa BSS{4}
    BSS{4, true, true}
    @test pick_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), zeros(99)) isa BSS{4}
    BSS{4, true, false}
    @test (
        @inferred pick_batchsize(
            AutoSimpleFiniteDiff(; chunksize = 4), @SVector(zeros(6))
        )
    ) isa BSS{4, false, false}
    @test (
        @inferred pick_batchsize(
            AutoSimpleFiniteDiff(; chunksize = 4), @SVector(zeros(100))
        )
    ) isa BSS{4, false, true}
end

@testset "Static arrays" begin
    # the batch size must be deduced from the type, for any static array shape
    for x in (@SVector(zeros(6)), @SMatrix(zeros(2, 3)), @MVector(zeros(6)))
        @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), x)) isa BSS{6, true, true}
        @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), x)) isa
            BSS{4, false, false}
        @test (@inferred pick_batchsize(AutoEnzyme(), x)) isa BSS{6, true, true}
        @test (@inferred prepare_jacobian(identity, AutoSimpleFiniteDiff(), x)) isa
            DI.PushforwardJacobianPrep{<:Any, BSS{6, true, true}}
        @test (
            @inferred prepare_jacobian(identity, AutoSimpleFiniteDiff(; chunksize = 4), x)
        ) isa DI.PushforwardJacobianPrep{<:Any, BSS{4, false, false}}
        @test (@inferred jacobian(identity, AutoSimpleFiniteDiff(), x)) isa StaticMatrix
    end

    # multiple batches must be concatenated without losing static sizes or inference
    f(x) = vcat(vec(x), vec(x) .^ 2)  # non-square Jacobian
    function f!(y, x)
        y[1:length(x)] .= vec(x)
        y[(length(x) + 1):end] .= vec(x) .^ 2
        return nothing
    end
    g(x) = sum(abs2, x) * sum(x)  # non-diagonal Hessian
    has_type(J, x) = x isa StaticArray ? J isa Union{SArray, MArray} : J isa Matrix
    # without preparation, the batch size is only inferred for static arrays
    function inferred_jacobian(f_or_f!y, backend, x)
        x isa StaticArray && @inferred jacobian(f_or_f!y..., backend, x)
        prep = prepare_jacobian(f_or_f!y..., backend, x)
        return @inferred jacobian(f_or_f!y..., prep, backend, x)
    end
    function inferred_hessian(f, backend, x)
        x isa StaticArray && @inferred hessian(f, backend, x)
        prep = prepare_hessian(f, backend, x)
        return @inferred hessian(f, prep, backend, x)
    end
    @testset "$(typeof(x))" for x in (
            @SVector(rand(6)),
            @SMatrix(rand(2, 3)),
            @MVector(rand(6)),
            rand(6),
            @SVector(rand(ComplexF64, 6)),
        )
        y = x isa StaticArray ? MVector{12, eltype(x)}(undef) : similar(x, 12)
        J_ref = jacobian(f, AutoSimpleFiniteDiff(), Array(x))
        # B = 1, aligned, unaligned and single batch use different methods
        @testset "Pushforward, chunksize $chunksize" for chunksize in (1, 3, 4, 6)
            backend = AutoSimpleFiniteDiff(; chunksize)
            J = inferred_jacobian((f,), backend, x)
            @test has_type(J, x)
            @test J ≈ J_ref
            J = inferred_jacobian((f!, y), backend, x)
            @test has_type(J, x)
            @test J ≈ J_ref
        end
        @testset "Pullback, chunksize $chunksize" for chunksize in (1, 4, 5, 12)
            backend = DI.AutoReverseFromPrimitive(AutoSimpleFiniteDiff(; chunksize))
            J = inferred_jacobian((f,), backend, x)
            @test has_type(J, x)
            @test J ≈ J_ref
            J = inferred_jacobian((f!, y), backend, x)
            @test has_type(J, x)
            @test J ≈ J_ref
        end
        if eltype(x) <: Real
            H_ref = 2 * (sum(x) * I + (vec(x) .+ vec(x)'))
            @testset "Hessian, chunksize $chunksize" for chunksize in (1, 3, 4, 6)
                backend = SecondOrder(
                    AutoSimpleFiniteDiff(; chunksize), AutoSimpleFiniteDiff()
                )
                H = inferred_hessian(g, backend, x)
                @test has_type(H, x)
                @test H ≈ H_ref rtol = 1.0e-4
            end
        end
    end
    @test_throws ArgumentError("Cannot concatenate zero batches") jacobian(
        identity, AutoSimpleFiniteDiff(; chunksize = 2), @SVector(zeros(0))
    )
end

@testset "Thresholding" begin
    @test threshold_batchsize(AutoSimpleFiniteDiff(), 2) isa AutoSimpleFiniteDiff{nothing}
    @test threshold_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), 2) isa
        AutoSimpleFiniteDiff{2}
    @test threshold_batchsize(AutoSimpleFiniteDiff(; chunksize = 4), 6) isa
        AutoSimpleFiniteDiff{4}
    @test threshold_batchsize(AutoSparse(AutoSimpleFiniteDiff(; chunksize = 4)), 2) isa
        AutoSparse{<:AutoSimpleFiniteDiff{2}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize = 4), AutoSimpleFiniteDiff(; chunksize = 3)
        ),
        6,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{4}, <:AutoSimpleFiniteDiff{3}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize = 4), AutoSimpleFiniteDiff(; chunksize = 3)
        ),
        2,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{2}, <:AutoSimpleFiniteDiff{2}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize = 1), AutoSimpleFiniteDiff(; chunksize = 3)
        ),
        2,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{1}, <:AutoSimpleFiniteDiff{2}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize = 4), AutoSimpleFiniteDiff(; chunksize = 1)
        ),
        2,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{2}, <:AutoSimpleFiniteDiff{1}}
    @test threshold_batchsize(
        MixedMode(AutoSimpleFiniteDiff(; chunksize = 4), AutoZygote()), 2
    ) isa MixedMode{<:AutoSimpleFiniteDiff{2}, <:AutoZygote}
end

@testset "Reasonable" begin
    for Bmax in 1:5
        @test all(<=(Bmax), reasonable_batchsize.(1:10, Bmax))
        @test issorted(div.(1:10, reasonable_batchsize.(1:10, Bmax), RoundUp))
        if Bmax > 2
            @test reasonable_batchsize(Bmax + 1, Bmax) < Bmax
        end
    end
end
