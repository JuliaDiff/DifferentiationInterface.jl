using DifferentiationInterface
using Enzyme: Enzyme
using Test

@testset "MutabilityError" begin
    f = let
        cache = [0.0]
        x -> sum(copyto!(cache, x))
    end

    msg = try
        gradient(f, AutoEnzyme(), [1.0])
    catch e
        buf = IOBuffer()
        showerror(buf, e)
        String(take!(buf))
    end
    @test occursin("AutoEnzyme", msg)
    @test occursin("function_annotation", msg)
    @test occursin("ADTypes", msg)
    @test occursin("DifferentiationInterface", msg)
end

@testset "RuntimeActivityError" begin
    function g(active_var, constant_var, cond)
        if cond
            return active_var
        else
            return constant_var
        end
    end

    function h(active_var, constant_var, cond)
        return [g(active_var, constant_var, cond), g(active_var, constant_var, cond)]
    end

    msg = try
        pushforward(
            h,
            AutoEnzyme(; mode=Enzyme.Forward),
            [1.0],
            ([1.0],),
            Constant([1.0]),
            Constant(true),
        )
    catch e
        buf = IOBuffer()
        showerror(buf, e)
        String(take!(buf))
    end
    @test occursin("AutoEnzyme", msg)
    @test occursin("mode", msg)
    @test occursin("set_runtime_activity", msg)
    @test occursin("ADTypes", msg)
    @test occursin("DifferentiationInterface", msg)
end
