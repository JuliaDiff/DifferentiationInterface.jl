make_empty(t::Number) = typeof(t)[]
function make_empty!(y::AbstractArray, t::Number)
    @assert isempty(y)
    return nothing
end

function empty_scenarios()
    scens = Scenario[
        Scenario{:derivative, :out}(make_empty, 1.0; res1 = Float64[]),
        Scenario{:derivative, :out}(make_empty!, Float64[], 1.0; res1 = Float64[]),
        Scenario{:gradient, :out}(sum, Float64[]; res1 = Float64[]),
        Scenario{:jacobian, :out}(copy, Float64[]; res1 = float.(I(0))),
        Scenario{:jacobian, :out}(copyto!, Float64[], Float64[]; res1 = float.(I(0))),
    ]
    return scens
end
