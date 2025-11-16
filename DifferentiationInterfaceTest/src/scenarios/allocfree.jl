function identity_scenarios(x::Number; dx::Number, dy::Number)
    f = identity
    dy_from_dx = dx
    dx_from_dy = dy
    der = oneunit(x)

    return [
        Scenario{:pushforward, :out}(f, x, (dx,); res1 = (dy_from_dx,)),
        Scenario{:pullback, :out}(f, x, (dy,); res1 = (dx_from_dy,)),
        Scenario{:derivative, :out}(f, x; res1 = der),
    ]
end

function sum_scenarios(x::AbstractArray; dx::AbstractArray, dy::Number)
    f = sum
    dy_from_dx = sum(dx)
    dx_from_dy = (similar(x) .= dy)
    grad = similar(x)
    grad .= oneunit(eltype(x))

    return [
        Scenario{:pushforward, :out}(f, x, (dx,); res1 = (dy_from_dx,)),
        Scenario{:pullback, :in}(f, x, (dy,); res1 = (dx_from_dy,)),
        Scenario{:gradient, :in}(f, x; res1 = grad),
    ]
end

function copy!_scenarios(x::AbstractArray; dx::AbstractArray, dy::AbstractArray)
    f! = copy!
    y = similar(x)
    f!(y, x)
    dy_from_dx = dx
    dx_from_dy = dy
    jac = Matrix(Diagonal(ones(eltype(x), length(x))))

    return [
        Scenario{:pushforward, :in}(f!, y, x, (dx,); res1 = (dy_from_dx,)),
        Scenario{:pullback, :in}(f!, y, x, (dy,); res1 = (dx_from_dy,)),
        Scenario{:jacobian, :in}(f!, y, x; res1 = jac),
    ]
end

"""
    allocfree_scenarios()

Create a vector of [`Scenario`](@ref)s with functions that do not allocate.

!!! warning

    At the moment, second-order scenarios are excluded.
"""
function allocfree_scenarios()
    x_ = 0.42
    dx_ = 3.14
    dy_ = -1 / 12

    x_6 = float.(1:6)
    dx_6 = float.(-1:-1:-6)
    dy_6 = float.(-5:2:5)

    scens = vcat(
        identity_scenarios(x_; dx = dx_, dy = dy_), #
        sum_scenarios(x_6; dx = dx_6, dy = dy_),
        copy!_scenarios(x_6; dx = dx_6, dy = dy_6),
    )
    return scens
end
