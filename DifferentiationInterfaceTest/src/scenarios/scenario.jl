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
struct Scenario{op,pl_op,pl_fun,F,X,Y,T<:Union{Nothing,NTuple},C<:Tuple,R1,R2}
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
    y = f(x, map(unwrap, contexts)...)
    return Scenario{op,pl_op,:out}(f; x, y, tang, contexts, res1, res2)
end

function Scenario{op,pl_op}(
    f!, y, x; tang=nothing, contexts=(), res1=nothing, res2=nothing
) where {op,pl_op}
    @assert op in ALL_OPS
    @assert pl_op in (:in, :out)
    return Scenario{op,pl_op,:in}(f!; x, y, tang, contexts, res1, res2)
end

Base.:(==)(scen1::Scenario, scen2::Scenario) = false

function Base.:(==)(
    scen1::Scenario{op,pl_op,pl_fun}, scen2::Scenario{op,pl_op,pl_fun}
) where {op,pl_op,pl_fun}
    eq_f = scen1.f == scen2.f
    eq_x = scen1.x == scen2.x
    eq_y = scen1.y == scen2.y
    eq_tang = scen1.tang == scen2.tang
    eq_contexts = all(
        map(scen1.contexts, scen2.contexts) do c1, c2
            if c1 isa Cache || c2 isa Cache
                return true
            else
                return c1 == c2
            end
        end,
    )
    eq_res1 = scen1.res1 == scen2.res1
    eq_res2 = scen1.res2 == scen2.res2
    return (eq_x && eq_y && eq_tang && eq_contexts && eq_res1 && eq_res2)
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
    place_compatible = function_place(scen) == :out || Bool(inplace_support(backend))
    sparse_compatible = operator(scen) in (:jacobian, :hessian) || !isa(backend, AutoSparse)
    secondorder_compatible =
        order(scen) == 2 || !isa(backend, Union{SecondOrder,AutoSparse{<:SecondOrder}})
    mixedmode_compatible =
        operator(scen) == :jacobian || !isa(backend, AutoSparse{<:MixedMode})
    return place_compatible &&
           secondorder_compatible &&
           sparse_compatible &&
           mixedmode_compatible
end

function group_by_operator(
    scenario_inds::AbstractVector{<:Integer}, scenarios::AbstractVector{<:Scenario}
)
    return Dict(
        op => filter(l -> operator(scenarios[l]) == op, scenario_inds) for
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
    if length(scen.contexts) > 0
        print(io, " ($(length(scen.contexts)) contexts)")
    end
    return nothing
end

function adapt_batchsize(backend::AbstractADType, scen::Scenario)
    if operator(scen) == :jacobian
        if ADTypes.mode(backend) isa Union{ADTypes.ForwardMode,ADTypes.ForwardOrReverseMode}
            return DI.threshold_batchsize(backend, length(scen.x))
        elseif ADTypes.mode(backend) isa ADTypes.ReverseMode
            return DI.threshold_batchsize(backend, length(scen.y))
        elseif ADTypes.mode(backend) isa DI.ForwardAndReverseMode
            return DI.threshold_batchsize(backend, min(length(scen.x), length(scen.y)))
        elseif ADTypes.mode(backend) isa ADTypes.SymbolicMode
            return backend
        else
            error("Unknown mode")
        end
    elseif operator(scen) == :hessian
        return DI.threshold_batchsize(backend, length(scen.x))
    else
        return backend
    end
end

function no_matrices(scens::AbstractVector{<:Scenario})
    return filter(s -> !isa(s.x, AbstractMatrix) && !isa(s.y, AbstractMatrix), scens)
end
