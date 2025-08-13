using Pkg
Pkg.add(["ChainRulesTestUtils", "FiniteDiff", "ForwardDiff", "Zygote", "Mooncake"])

using ChainRulesTestUtils: ChainRulesTestUtils
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Mooncake: Mooncake
using StableRNGs
using Test

LOGGING = get(ENV, "CI", "false") == "false"

struct ADBreaker{F}
    f::F
end

function (adb::ADBreaker)(x::Number)
    copyto!(Float64[0], x)  # break ForwardDiff and Zygote
    return adb.f(x)
end

function (adb::ADBreaker)(x::AbstractArray)
    copyto!(similar(x, Float64), x)  # break ForwardDiff and Zygote
    return adb.f(x)
end

function differentiatewith_scenarios()
    outofplace_scens = filter(DIT.default_scenarios()) do scen
        DIT.function_place(scen) == :out
    end
    # with bad_scens, everything would break
    bad_scens = map(outofplace_scens) do scen
        DIT.change_function(scen, ADBreaker(scen.f))
    end
    # with good_scens, everything is fixed
    good_scens = map(bad_scens) do scen
        DIT.change_function(scen, DifferentiateWith(scen.f, AutoFiniteDiff()))
    end
    return good_scens
end

test_differentiation(
    [AutoForwardDiff(), AutoZygote(), AutoMooncake(; config=nothing)],
    differentiatewith_scenarios();
    excluded=SECOND_ORDER,
    logging=LOGGING,
    testset_name="DI tests",
)

@testset "ChainRules tests" begin
    @testset for scen in filter(differentiatewith_scenarios()) do scen
        DIT.operator(scen) == :pullback
    end
        ChainRulesTestUtils.test_rrule(scen.f, scen.x; rtol=1e-4)
    end
end;

@testset "Mooncake tests" begin
    @testset for scen in filter(differentiatewith_scenarios()) do scen
        DIT.operator(scen) == :pullback
    end
        Mooncake.TestUtils.test_rule(
            StableRNG(0), scen.f, scen.x; is_primitive=true, mode=Mooncake.ReverseMode()
        )
    end
end;

@testset "Mooncake errors" begin
    MooncakeDifferentiateWithError =
        Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceMooncakeExt).MooncakeDifferentiateWithError

    e = MooncakeDifferentiateWithError(identity, 1.0, 2.0)
    @test sprint(showerror, e) ==
        "MooncakeDifferentiateWithError: For the function type typeof(identity) and input type Float64, the output type Float64 is currently not supported."

    f_num2tup(x::Number) = (x,)
    f_vec2tup(x::Vector) = (first(x),)
    f_tup2num(x::Tuple{<:Number}) = only(x)
    f_tup2vec(x::Tuple{<:Number}) = [only(x)]

    @test_throws MooncakeDifferentiateWithError pullback(
        DifferentiateWith(f_num2tup, AutoFiniteDiff()),
        AutoMooncake(; config=nothing),
        1.0,
        ((2.0,),),
    )
    @test_throws MooncakeDifferentiateWithError pullback(
        DifferentiateWith(f_vec2tup, AutoFiniteDiff()),
        AutoMooncake(; config=nothing),
        [1.0],
        ((2.0,),),
    )
    @test_throws MethodError pullback(
        DifferentiateWith(f_tup2num, AutoFiniteDiff()),
        AutoMooncake(; config=nothing),
        (1.0,),
        (2.0,),
    )
    @test_throws MethodError pullback(
        DifferentiateWith(f_tup2vec, AutoFiniteDiff()),
        AutoMooncake(; config=nothing),
        (1.0,),
        ([2.0],),
    )
end
