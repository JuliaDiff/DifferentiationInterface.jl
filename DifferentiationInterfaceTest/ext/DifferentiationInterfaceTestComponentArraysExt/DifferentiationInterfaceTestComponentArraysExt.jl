module DifferentiationInterfaceTestComponentArraysExt

using ComponentArrays: ComponentVector
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using LinearAlgebra: dot

## Vector to scalar

function comp_to_num(x::ComponentVector)::Number
    return sum(sin.(x.a)) + sum(cos.(x.b))
end

comp_to_num_gradient(x) = ComponentVector(; a=cos.(x.a), b=-sin.(x.b))

function comp_to_num_pushforward(x, dx)
    g = comp_to_num_gradient(x)
    return dot(g, dx)
end

function comp_to_num_pullback(x, dy)
    return comp_to_num_gradient(x) .* dy
end

function comp_to_num_scenarios_onearg(x::ComponentVector; dx::AbstractVector, dy::Number)
    f = comp_to_num
    dy_from_dx = comp_to_num_pushforward(x, dx)
    dx_from_dy = comp_to_num_pullback(x, dy)
    grad = comp_to_num_gradient(x)

    # pushforward stays out of place
    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                Scenario{:pullback,pl_op}(f, x; tang=(dy,), res1=(dx_from_dy,)),
                Scenario{:gradient,pl_op}(f, x; res1=grad),
            ],
        )
    end
    for pl_op in (:out,)
        append!(scens, [Scenario{:pushforward,pl_op}(f, x; tang=(dx,), res1=(dy_from_dx,))])
    end
    return scens
end

## Gather

function DIT.component_scenarios()
    dy_ = -1 / 12

    x_comp = ComponentVector(; a=float.(1:4), b=float.(5:6))
    dx_comp = ComponentVector(; a=float.(4:-1:1), b=float.(6:-1:5))

    scens = vcat(
        # one argument
        comp_to_num_scenarios_onearg(x_comp::ComponentVector; dx=dx_comp, dy=dy_),
        # two arguments
    )
    return scens
end

end
