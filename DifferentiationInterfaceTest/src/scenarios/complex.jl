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

    dy_6 = float.(-5:2:5) .+ im
    dy_12 = float.(-11:2:11) .+ im

    V = Vector{Complex{Float64}}

    scens = vcat(
        # one argument
        num_to_num_scenarios(x_; dx=dx_, dy=dy_, add_vectors=false),
        num_to_arr_scenarios_onearg(x_, V; dx=dx_, dy=dy_6),
        arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_),
        vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, V; dx=dx_, dy=dy_6),
        vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
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
