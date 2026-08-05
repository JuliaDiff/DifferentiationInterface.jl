@kwdef struct Scenario{O, F, A1 <: Tuple, A2 <: Tuple, R <: Result}
    operator::O
    func::F
    pre_backend_args::A1
    post_backend_args::A2
    result::R
    name::Union{String, Nothing} = nothing
end

operator(s::Scenario) = s.operator
func(s::Scenario) = s.func
pre_backend_args(s::Scenario) = s.pre_backend_args
post_backend_args(s::Scenario) = s.post_backend_args
result(s::Scenario) = s.result
name(s::Scenario) = s.name

order(s::Scenario) = order(operator(s))
is_inplace(s::Scenario) = is_inplace(operator(s))
has_tangent(s::Scenario) = needs_tangent(operator(s))

function has_inplace_function(s::Scenario)
    op = operator(s)
    stdop = standard(op)
    if stdop in (gradient, hvp, hessian)
        return false
    elseif order(op) == 1
        if is_inplace(op)
            return length(pre_backend_args(s)) == 2
        else
            return length(pre_backend_args(s)) == 1
        end
    else
        if is_inplace(op)
            return length(pre_backend_args(s)) == 3
        else
            return length(pre_backend_args(s)) == 2
        end
    end
end

function value_args(s::Scenario)
    if has_inplace_function(s)
        return (first(pre_backend_args(s)),)
    else
        return ()
    end
end

clean_pre_backend_args(s::Scenario) = map(a -> fmap(similar, a), pre_backend_args(s))
clean_value_args(s::Scenario) = map(a -> fmap(similar, a), value_args(s))
protected_post_backend_args(s::Scenario) = filter(a -> !(a isa Union{Cache, ConstantOrCache}), post_backend_args(s))

function Base.show(io::IO, s::Scenario)
    !isnothing(s.name) && print(io, "Scenario ", s.name, ":\n")
    print(io, operator(s), "(")
    print(io, "::", repr(typeof(func(s))), ", ")
    for a in pre_backend_args(s)
        print(io, "::", repr(typeof(a)), ", ")
    end
    print(io, "::AbstractADType, ")
    for (i, a) in enumerate(post_backend_args(s))
        print(io, "::", repr(typeof(a)))
        if i < length(post_backend_args(s))
            print(io, ", ")
        end
    end
    print(io, ")")
    return nothing
end

@kwdef struct TripleScenario{S <: Scenario}
    prep_scen::S
    exec_scen::S
    flush_scen::S
end

function TripleScenario(s::Scenario)
    # by default, prepare and flush on zeroed-out scenario
    prep_scen = Scenario(
        operator(s),
        deepcopy(func(s)),
        map(a -> fmap(myzero, a), pre_backend_args(s)),
        map(a -> fmap(myzero, a), post_backend_args(s)),
        fmap(myzero, result(s)),
        isnothing(name(s)) ? nothing : string(name(s), " - zero")
    )
    exec_scen = s
    flush_scen = deepcopy(prep_scen)
    return TripleScenario(prep_scen, exec_scen, flush_scen)
end

operator(ts::TripleScenario) = operator(ts.exec_scen)

function Base.show(io::IO, ts::TripleScenario)
    return print(io, ts.exec_scen)
end

function result_from_input(s::Scenario, input)
    op = operator(s)
    input_order0 = if has_inplace_function(s)
        input[1]
    else
        nothing
    end
    input_order1 = if is_inplace(op) && (order(s) == 1 || has_value(op))
        input[1 + !isnothing(input_order0)]
    else
        nothing
    end
    input_order2 = if is_inplace(op) && order(s) == 2
        input[1 + !isnothing(input_order0) + !isnothing(input_order1)]
    else
        nothing
    end
    return Result(input_order0, input_order1, input_order2)
end

function result_from_output(s::Scenario, output)
    op = operator(s)
    return if order(op) == 1
        if has_value(op)
            Result(output[1], output[2], nothing)
        else
            Result(nothing, output, nothing)
        end
    else
        if has_value(op)
            if standard(op) == hvp
                # special case, value not returned
                Result(nothing, output[1], output[2])
            else
                Result(output[1], output[2], output[3])
            end
        else
            Result(nothing, nothing, output)
        end
    end
end

function should_reprepare(ts::TripleScenario)
    (; prep_scen, exec_scen) = ts
    x_prep = first(post_backend_args(prep_scen))
    x_exec = first(post_backend_args(exec_scen))
    return hasmethod(size, (typeof(x_prep),)) && hasmethod(size, (typeof(x_exec),)) && size(x_prep) != size(x_exec)
end

function adapt_batchsize(backend::AbstractADType, ts::TripleScenario)
    (; prep_scen, exec_scen) = ts
    x_prep = first(post_backend_args(prep_scen))
    x_exec = first(post_backend_args(exec_scen))
    y_prep = order0(result(prep_scen))
    y_exec = order0(result(exec_scen))
    Bmax = if x_exec isa AbstractArray && y_exec isa AbstractArray
        min(length(x_exec), length(y_exec), length(x_prep), length(y_prep))
    elseif x_exec isa AbstractArray
        min(length(x_exec), length(x_prep))
    elseif y_exec isa AbstractArray
        min(length(y_exec), length(y_prep))
    else
        typemax(Int)
    end
    return DI.threshold_batchsize(backend, Bmax)
end
