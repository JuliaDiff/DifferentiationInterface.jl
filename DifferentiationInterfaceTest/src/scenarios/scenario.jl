const ALL_OPS = (
    :pushforward,
    :pullback,
    :derivative,
    :gradient,
    :jacobian,
    :hessian,
    :hvp,
    :second_derivative,
)

"""
    Scenario{op,pl_op,pl_fun}

Store a testing scenario composed of a function and its input + output for a given operator.

This generic type should never be used directly: use the specific constructor corresponding to the operator you want to test, or a predefined list of scenarios.

# Type parameters

- `op`: one  of `:pushforward`, `:pullback`, `:derivative`, `:gradient`, `:jacobian`,`:second_derivative`, `:hvp`, `:hessian`
- `pl_op`: either `:in` (for `op!(f, result, backend, x)`) or `:out` (for `result = op(f, backend, x)`)
- `pl_fun`: either `:in` (for `f!(y, x)`) or `:out` (for `y = f(x)`)

# Constructors

    Scenario{op,pl_op}(f, x; tang, contexts, res1, res2)
    Scenario{op,pl_op}(f!, y, x; tang, contexts, res1, res2)

# Fields

$(TYPEDFIELDS)
"""
struct Scenario{op,pl_op,pl_fun,F,X,Y,T<:Union{Nothing,Tangents},C<:Tuple,R1,R2}
    "function `f` (if `args==1`) or `f!` (if `args==2`) to apply"
    f::F
    "primal input"
    x::X
    "primal output"
    y::Y
    "tangents for pushforward, pullback or HVP"
    tang::T
    "contexts (if applicable)"
    contexts::C
    "first-order result of the operator (if applicable)"
    res1::R1
    "second-order result of the operator (if applicable)"
    res2::R2
end

function Scenario{op,pl_op,pl_fun}(
    f::F; x::X, y::Y, tang::T, contexts::C, res1::R1, res2::R2
) where {op,pl_op,pl_fun,F,X,Y,T,C,R1,R2}
    return Scenario{op,pl_op,pl_fun,F,X,Y,T,C,R1,R2}(f, x, y, tang, contexts, res1, res2)
end

function Scenario{op,pl_op}(
    f, x; tang=nothing, contexts=(), res1=nothing, res2=nothing
) where {op,pl_op}
    @assert op in ALL_OPS
    @assert pl_op in (:in, :out)
    y = f(x)
    return Scenario{op,pl_op,:out}(f; x, y, tang, contexts, res1, res2)
end

function Scenario{op,pl_op}(
    f!, y, x; tang=nothing, contexts=(), res1=nothing, res2=nothing
) where {op,pl_op}
    @assert op in ALL_OPS
    @assert pl_op in (:in, :out)
    f!(y, x)
    return Scenario{op,pl_op,:in}(f!; x, y, tang, contexts, res1, res2)
end

operator(::Scenario{op}) where {op} = op
operator_place(::Scenario{op,pl_op}) where {op,pl_op} = pl_op
function_place(::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun} = pl_fun

function order(scen::Scenario)
    if operator(scen) in [:pushforward, :pullback, :derivative, :gradient, :jacobian]
        return 1
    elseif operator(scen) in [:hvp, :hessian, :second_derivative]
        return 2
    end
end

function compatible(backend::AbstractADType, scen::Scenario)
    return function_place(scen) == :out || Bool(inplace_support(backend))
end

function group_by_operator(scenarios::AbstractVector{<:Scenario})
    return Dict(
        op => filter(s -> operator(s) == op, scenarios) for
        op in unique(operator.(scenarios))
    )
end

function Base.show(
    io::IO, scen::Scenario{op,pl_op,pl_fun,F,X,Y,T}
) where {op,pl_op,pl_fun,F,X,Y,T}
    print(io, "Scenario{$(repr(op)),$(repr(pl_op))} $(string(scen.f)) : $X -> $Y")
    if op in (:pushforward, :pullback, :hvp)
        print(io, " ($(length(scen.tang)) tangents)")
    end
    return nothing
end
