using ADTypes
using DifferentiationInterface
using DifferentiationInterface:
    AutoSimpleFiniteDiff,
    BatchSizeSettings,
    pick_batchsize,
    reasonable_batchsize,
    threshold_batchsize
import DifferentiationInterface as DI
using StaticArrays
using Test

BSS = BatchSizeSettings

@testset "Default" begin
    @test (@inferred pick_batchsize(AutoZygote(), zeros(2))) isa BSS{1,false,true}
    @test (@inferred pick_batchsize(AutoZygote(), zeros(100))) isa BSS{1,false,true}
    @test_throws ArgumentError pick_batchsize(AutoSparse(AutoZygote()), zeros(2))
    @test_throws ArgumentError pick_batchsize(
        SecondOrder(AutoZygote(), AutoZygote()), zeros(2)
    )
    @test_throws ArgumentError pick_batchsize(
        MixedMode(AutoSimpleFiniteDiff(), AutoZygote()), zeros(2)
    )
end

@testset "SimpleFiniteDiff (adaptive)" begin
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(0))) isa BSS{0,true,true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(2))) isa BSS{2,true,true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(6))) isa BSS{6,true,true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(12))) isa BSS{12,true,true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(24))) isa BSS{12,false,true}
    @test (pick_batchsize(AutoSimpleFiniteDiff(), zeros(100))) isa BSS{12,false,false}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(0)))) isa
        BSS{0,true,true}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(2)))) isa
        BSS{2,true,true}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(6)))) isa
        BSS{6,true,true}
    @test (@inferred pick_batchsize(AutoSimpleFiniteDiff(), @SVector(zeros(100)))) isa
        BSS{100,true,true}
end

@testset "SimpleFiniteDiff (fixed)" begin
    @test_throws ArgumentError pick_batchsize(AutoSimpleFiniteDiff(; chunksize=4), zeros(2))
    @test_throws ArgumentError pick_batchsize(
        AutoSimpleFiniteDiff(; chunksize=4), @SVector(zeros(2))
    )
    @test pick_batchsize(AutoSimpleFiniteDiff(; chunksize=4), zeros(6)) isa BSS{4}
    @test pick_batchsize(AutoSimpleFiniteDiff(; chunksize=4), zeros(100)) isa BSS{4}
    BSS{4,true,true}
    @test pick_batchsize(AutoSimpleFiniteDiff(; chunksize=4), zeros(99)) isa BSS{4}
    BSS{4,true,false}
    @test (@inferred pick_batchsize(
        AutoSimpleFiniteDiff(; chunksize=4), @SVector(zeros(6))
    )) isa BSS{4,false,false}
    @test (@inferred pick_batchsize(
        AutoSimpleFiniteDiff(; chunksize=4), @SVector(zeros(100))
    )) isa BSS{4,false,true}
end

@testset "Thresholding" begin
    @test threshold_batchsize(AutoSimpleFiniteDiff(), 2) isa AutoSimpleFiniteDiff{nothing}
    @test threshold_batchsize(AutoSimpleFiniteDiff(; chunksize=4), 2) isa
        AutoSimpleFiniteDiff{2}
    @test threshold_batchsize(AutoSimpleFiniteDiff(; chunksize=4), 6) isa
        AutoSimpleFiniteDiff{4}
    @test threshold_batchsize(AutoSparse(AutoSimpleFiniteDiff(; chunksize=4)), 2) isa
        AutoSparse{<:AutoSimpleFiniteDiff{2}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize=4), AutoSimpleFiniteDiff(; chunksize=3)
        ),
        6,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{4},<:AutoSimpleFiniteDiff{3}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize=4), AutoSimpleFiniteDiff(; chunksize=3)
        ),
        2,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{2},<:AutoSimpleFiniteDiff{2}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize=1), AutoSimpleFiniteDiff(; chunksize=3)
        ),
        2,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{1},<:AutoSimpleFiniteDiff{2}}
    @test threshold_batchsize(
        SecondOrder(
            AutoSimpleFiniteDiff(; chunksize=4), AutoSimpleFiniteDiff(; chunksize=1)
        ),
        2,
    ) isa SecondOrder{<:AutoSimpleFiniteDiff{2},<:AutoSimpleFiniteDiff{1}}
    @test threshold_batchsize(
        MixedMode(AutoSimpleFiniteDiff(; chunksize=4), AutoZygote()), 2
    ) isa MixedMode{<:AutoSimpleFiniteDiff{2},<:AutoZygote}
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
