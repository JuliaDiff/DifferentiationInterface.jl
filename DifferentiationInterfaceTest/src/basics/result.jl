@kwdef struct Result{O0, O1, O2}
    order0::O0 = nothing
    order1::O1 = nothing
    order2::O2 = nothing
end

order0(r::Result) = r.order0
order1(r::Result) = r.order1
order2(r::Result) = r.order2

function test_approx(
        op::Function,
        computed::Result, reference::Result;
        isapprox::Function, atol::Real, rtol::Real
    )
    if has_value(op) && standard(op) != hvp
        @testset "Primal" begin
            @test isapprox(order0(computed), order0(reference); atol, rtol)
        end
    end
    if order(op) == 1 || has_value(op)
        @testset "First order" begin
            if order(op) == 1 && needs_tangent(op)
                foreach(order1(computed), order1(reference)) do tang_computed, tang_ref
                    @test isapprox(tang_computed, tang_ref; atol, rtol)
                end
            else
                @test isapprox(order1(computed), order1(reference); atol, rtol)
            end
        end
    end
    if order(op) == 2
        @testset "Second order" begin
            if needs_tangent(op)
                foreach(order2(computed), order2(reference)) do tang_computed, tang_ref
                    @test isapprox(tang_computed, tang_ref; atol, rtol)
                end
            else
                @test isapprox(order2(computed), order2(reference); atol, rtol)
            end
        end
    end
    return nothing
end

function test_same(
        op::Function, output::Result, input::Result
    )
    if has_value(op) && !isnothing(order0(input))
        @test order0(output) === order0(input)
    end
    if (order(op) == 1 || has_value(op)) && !isnothing(order1(input))
        @test order1(output) === order1(input)
    end
    if order(op) == 2 && !isnothing(order2(input))
        @test order2(output) === order2(input)
    end
    return nothing
end
