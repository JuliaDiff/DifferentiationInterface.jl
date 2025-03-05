square_only(x::AbstractVector) = only(x)^2
abs2_only(x::AbstractVector) = abs2(only(x))

function complex_holomorphic_gradient_scenarios()
    # http://arxiv.org/abs/2409.06752
    dy = 1.0
    x = [1.0 + im]
    grad = 2 * conj(x)
    scens = Scenario[
        Scenario{:gradient,:out}(square_only, x; res1=grad),
        Scenario{:gradient,:in}(square_only, x; res1=grad),
        Scenario{:pullback,:out}(square_only, x; tang=(dy,), res1=(grad,)),
        Scenario{:pullback,:in}(square_only, x; tang=(dy,), res1=(grad,)),
    ]
    return scens
end

function complex_gradient_scenarios()
    dy = 1.0
    x = [1.0 + im]
    grad = 2 * x
    scens = Scenario[
        Scenario{:gradient,:out}(abs2_only, x; res1=grad),
        Scenario{:gradient,:in}(abs2_only, x; res1=grad),
        Scenario{:pullback,:out}(abs2_only, x; tang=(dy,), res1=(grad,)),
        Scenario{:pullback,:in}(abs2_only, x; tang=(dy,), res1=(grad,)),
    ]
    return scens
end

"""
    complex_scenarios()

Create a vector of first-order [`Scenario`](@ref)s with complex-valued array types and holomorphic functions only.
"""
function complex_scenarios()
    x_ = 0.42 + im
    dx_ = 3.14 + im
    dy_ = -1 / 12 + im

    x_6 = float.(1:6) .+ im
    dx_6 = float.(-1:-1:-6) .+ im

    dy_2 = [-3.0, 4.0] .+ im
    dy_6 = float.(-5:2:5) .+ im
    dy_12 = float.(-11:2:11) .+ im

    scens = vcat(
        # one argument
        num_to_num_scenarios(x_; dx=dx_, dy=dy_),
        num_to_vec_scenarios_onearg(x_; dx=dx_, dy=dy_2),
        arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_),
        vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        # two arguments
        num_to_vec_scenarios_twoarg(x_; dx=dx_, dy=dy_6),
        vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
        # complex gradients
        complex_gradient_scenarios(),
        complex_holomorphic_gradient_scenarios(),
    )

    return filter(s -> !(operator(s) in SECOND_ORDER), scens)
end

"""
    complex_sparse_scenarios()

Create a vector of Jacobian [`Scenario`](@ref)s with complex-valued array types and holomorphic functions only.
"""
function complex_sparse_scenarios()
    x_6 = float.(1:6) .+ float.(1:6) .^ 2 .* im
    scens = sparse_vec_to_vec_scenarios(x_6)
    return scens
end
