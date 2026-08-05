abstract type FunctionModifier end

"""
    zero(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the first- and second-order results which are set to zero.
"""
function Base.zero(scen::Scenario{op, pl_op, pl_fun}) where {op, pl_op, pl_fun}
    zero_res1 = if op in (:pushforward, :pullback)
        map(zero, scen.res1)
    else
        zero(scen.res1)
    end
    zero_res2 = if isnothing(scen.res2)
        nothing
    elseif op == :hvp
        map(zero, scen.res2)
    else
        zero(scen.res2)
    end
    return Scenario{op, pl_op, pl_fun}(;
        f = scen.f,
        x = scen.x,
        y = scen.y,
        t = scen.t,
        contexts = scen.contexts,
        res1 = zero_res1,
        res2 = zero_res2,
        prep_args = scen.prep_args,
        name = isnothing(scen.name) ? nothing : scen.name * " [zero]",
    )
end

"""
    change_function(scen::Scenario, new_f)

Return a new `Scenario` identical to `scen` except for the function `f` which is changed to `new_f`.
"""
function change_function(scen::Scenario{op, pl_op, pl_fun}, new_f) where {op, pl_op, pl_fun}
    return Scenario{op, pl_op, pl_fun}(;
        f = new_f,
        x = scen.x,
        y = scen.y,
        t = scen.t,
        contexts = scen.contexts,
        res1 = scen.res1,
        res2 = scen.res2,
        prep_args = scen.prep_args,
        name = isnothing(scen.name) ? nothing : scen.name * " [new function]",
    )
end

same_function(scen) = change_function(scen, scen.f)

"""
    batchify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the tangents `tang` and associated results `res1` / `res2`, which are duplicated (batch mode).

Only works if `scen` is a `pushforward`, `pullback` or `hvp` scenario.
"""
function batchify(scen::Scenario{op, pl_op, pl_fun}) where {op, pl_op, pl_fun}
    (; f, x, y, t, contexts, res1, res2, prep_args) = scen
    new_t = (only(t), -only(t))
    new_prep_args = if pl_fun == :out
        (;
            x = prep_args.x,
            contexts = prep_args.contexts,
            t = (only(prep_args.t), -only(prep_args.t)),
        )
    else
        (;
            y = prep_args.y,
            x = prep_args.x,
            contexts = prep_args.contexts,
            t = (only(prep_args.t), -only(prep_args.t)),
        )
    end
    if op == :pushforward || op == :pullback
        new_res1 = (only(res1), -only(res1))
        return Scenario{op, pl_op, pl_fun}(;
            f,
            x,
            y,
            t = new_t,
            contexts,
            res1 = new_res1,
            res2,
            prep_args = new_prep_args,
            name = isnothing(scen.name) ? nothing : scen.name * " [batchified]",
        )
    elseif op == :hvp
        new_res2 = (only(res2), -only(res2))
        return Scenario{op, pl_op, pl_fun}(;
            f,
            x,
            y,
            t = new_t,
            contexts,
            res1,
            res2 = new_res2,
            prep_args = new_prep_args,
            name = isnothing(scen.name) ? nothing : scen.name * " [batchified]",
        )
    end
end

struct WritableClosure{pl_fun, F, X, Y} <: FunctionModifier
    f::F
    x_buffer::Vector{X}
    y_buffer::Vector{Y}
    a::Float64
    b::Vector{Float64}
end

function WritableClosure{pl_fun}(
        f::F, x_buffer::Vector{X}, y_buffer::Vector{Y}, a, b
    ) where {pl_fun, F, X, Y}
    return WritableClosure{pl_fun, F, X, Y}(f, x_buffer, y_buffer, a, b)
end

Base.show(io::IO, f::WritableClosure) = print(io, "WritableClosure($(f.f))")

function (mc::WritableClosure{:out})(x)
    (; f, x_buffer, y_buffer, a, b) = mc
    x_buffer[1] = copy(x)
    y_buffer[1] = (a + only(b)) * f(x)
    return copy(y_buffer[1])
end

function (mc::WritableClosure{:in})(y, x)
    (; f, x_buffer, y_buffer, a, b) = mc
    x_buffer[1] = copy(x)
    f(y_buffer[1], x_buffer[1])
    y_buffer[1] .*= (a + only(b))
    copyto!(y, y_buffer[1])
    return nothing
end

"""
    closurify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the function `f` which is made to close over differentiable data.
"""
function closurify(scen::Scenario{op, pl_op, pl_fun}) where {op, pl_op, pl_fun}
    (; f, x, y) = scen
    @assert isempty(scen.contexts)
    x_buffer = [zero(x)]
    y_buffer = [zero(y)]
    a = 3.0
    b = [4.0]
    closure_f = WritableClosure{pl_fun}(f, x_buffer, y_buffer, a, b)
    return Scenario{op, pl_op, pl_fun}(;
        f = closure_f,
        x = scen.x,
        y = mymultiply(scen.y, a + only(b)),
        t = scen.t,
        contexts = scen.contexts,
        res1 = mymultiply(scen.res1, a + only(b)),
        res2 = mymultiply(scen.res2, a + only(b)),
        prep_args = scen.prep_args,
        name = isnothing(scen.name) ? nothing : scen.name * " [closurified]",
    )
end

struct MultiplyByConstant{pl_fun, F} <: FunctionModifier
    f::F
end

MultiplyByConstant{pl_fun}(f::F) where {pl_fun, F} = MultiplyByConstant{pl_fun, F}(f)

Base.show(io::IO, f::MultiplyByConstant) = print(io, "MultiplyByConstant($(f.f))")

function (mc::MultiplyByConstant{:out})(x, a)
    y = a * mc.f(x)
    return y
end

function (mc::MultiplyByConstant{:in})(y, x, a)
    mc.f(y, x)
    y .*= a
    return nothing
end

"""
    constantify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the function `f`, which is made to accept an additional constant argument by which the output is multiplied.
The output and result fields are updated accordingly.
"""
function constantify(scen::Scenario{op, pl_op, pl_fun}) where {op, pl_op, pl_fun}
    (; f) = scen
    @assert isempty(scen.contexts)
    multiply_f = MultiplyByConstant{pl_fun}(f)
    a = 3.0
    return Scenario{op, pl_op, pl_fun}(;
        f = multiply_f,
        x = scen.x,
        y = mymultiply(scen.y, a),
        t = scen.t,
        contexts = (Constant(a),),
        res1 = mymultiply(scen.res1, a),
        res2 = mymultiply(scen.res2, a),
        prep_args = (; scen.prep_args..., contexts = (Constant(-a),)),
        name = isnothing(scen.name) ? nothing : scen.name * " [constantified]",
    )
end

struct StoreInCache{pl_fun, F} <: FunctionModifier
    f::F
end

function StoreInCache{pl_fun}(f::F) where {pl_fun, F}
    return StoreInCache{pl_fun, F}(f)
end

Base.show(io::IO, f::StoreInCache) = print(io, "StoreInCache($(f.f))")

(sc::StoreInCache{:out})(x, y_cache::NamedTuple) = sc(x, y_cache.useful_cache)
(sc::StoreInCache{:in})(y, x, y_cache::NamedTuple) = sc(y, x, y_cache.useful_cache)
(sc::StoreInCache{:out})(x, y_cache::Tuple) = sc(x, first(y_cache))
(sc::StoreInCache{:in})(y, x, y_cache::Tuple) = sc(y, x, first(y_cache))

function (sc::StoreInCache{:out})(x, y_cache)  # no annotation otherwise Zygote.Buffer cries
    y = sc.f(x)
    if y isa Number
        y_cache[1] = y
        return y_cache[1]
    else
        copyto!(y_cache, y)
        return copy(y_cache)
    end
end

function (sc::StoreInCache{:in})(y, x, y_cache)
    sc.f(y_cache, x)
    copyto!(y, y_cache)
    return nothing
end

"""
    cachify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the function `f`, which is made to accept an additional cache argument to store the result before it is returned.

If `tup=true` the cache is a tuple of arrays, otherwise just an array.
"""
function cachify(scen::Scenario{op, pl_op, pl_fun}; use_tuples) where {op, pl_op, pl_fun}
    (; f) = scen
    @assert isempty(scen.contexts)
    cache_f = StoreInCache{pl_fun}(f)
    if use_tuples
        y_cache = if scen.y isa Number
            (; useful_cache = ([zero(scen.y)],), useless_cache = [zero(scen.y)])
        else
            (; useful_cache = (similar(scen.y),), useless_cache = similar(scen.y))
        end
    else
        y_cache = if scen.y isa Number
            [zero(scen.y)]
        else
            similar(scen.y)
        end
    end
    return Scenario{op, pl_op, pl_fun}(;
        f = cache_f,
        x = scen.x,
        y = scen.y,
        t = scen.t,
        contexts = (Cache(y_cache),),
        res1 = scen.res1,
        res2 = scen.res2,
        prep_args = (; scen.prep_args..., contexts = (Cache(y_cache),)),
        name = isnothing(scen.name) ? nothing : scen.name * " [cachified]",
    )
end

struct MultiplyByConstantAndStoreInCache{pl_fun, F} <: FunctionModifier
    f::F
end

function MultiplyByConstantAndStoreInCache{pl_fun}(f::F) where {pl_fun, F}
    return MultiplyByConstantAndStoreInCache{pl_fun, F}(f)
end

function Base.show(io::IO, f::MultiplyByConstantAndStoreInCache)
    return print(io, "MultiplyByConstantAndStoreInCache($(f.f))")
end

function (sc::MultiplyByConstantAndStoreInCache{:out})(x, constantorcache)
    (; constant, cache) = constantorcache
    (; a, b) = constant
    y = (a + only(b)) * sc.f(x)
    if eltype(y) == eltype(cache)
        newcache = cache
    else
        # poor man's PreallocationTools
        newcache = similar(cache, eltype(y))
    end
    if y isa Number
        newcache[1] = y
        return newcache[1]
    else
        copyto!(newcache, y)
        return copy(newcache)
    end
end

function (sc::MultiplyByConstantAndStoreInCache{:in})(y, x, constantorcache)
    (; constant, cache) = constantorcache
    (; a, b) = constant
    if eltype(y) == eltype(cache)
        newcache = cache
    else
        # poor man's PreallocationTools
        newcache = similar(cache, eltype(y))
    end
    sc.f(newcache, x)
    newcache .*= (a + only(b))
    copyto!(y, newcache)
    return nothing
end

"""
    constantorcachify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the function `f`, which is made to accept an additional "constant or cache" argument.
"""
function constantorcachify(scen::Scenario{op, pl_op, pl_fun}) where {op, pl_op, pl_fun}
    (; f) = scen
    @assert isempty(scen.contexts)
    constantorcache_f = MultiplyByConstantAndStoreInCache{pl_fun}(f)
    a = 3.0
    b = [4.0]
    constantorcache = if scen.y isa Number
        (; cache = [zero(scen.y)], constant = (; a, b))
    else
        (; cache = similar(scen.y), constant = (; a, b))
    end
    prep_constantorcache = if scen.y isa Number
        (; cache = [zero(scen.y)], constant = (; a = 2a, b = 3b))
    else
        (; cache = similar(scen.y), constant = (; a = 2a, b = 3b))
    end
    return Scenario{op, pl_op, pl_fun}(;
        f = constantorcache_f,
        x = scen.x,
        y = mymultiply(scen.y, a + only(b)),
        t = scen.t,
        contexts = (ConstantOrCache(constantorcache),),
        res1 = mymultiply(scen.res1, a + only(b)),
        res2 = mymultiply(scen.res2, a + only(b)),
        prep_args = (; scen.prep_args..., contexts = (ConstantOrCache(prep_constantorcache),)),
        name = isnothing(scen.name) ? nothing : scen.name * " [constantorcachified]",
    )
end

## Group functions

function batchify(scens::AbstractVector{<:Scenario})
    batchifiable_scens = filter(s -> operator(s) in (:pushforward, :pullback, :hvp), scens)
    return batchify.(batchifiable_scens)
end

closurify(scens::AbstractVector{<:Scenario}) = closurify.(scens)
constantify(scens::AbstractVector{<:Scenario}) = constantify.(scens)
cachify(scens::AbstractVector{<:Scenario}; use_tuples) = cachify.(scens; use_tuples)
constantorcachify(scens::AbstractVector{<:Scenario}) = constantorcachify.(scens)

## Compute results with backend

get_res1(::Val, args...) = nothing
get_res2(::Val, args...) = nothing

function get_res1(::Val{:derivative}, f, backend::AbstractADType, x, contexts...)
    return derivative(f, backend, x, contexts...)
end
function get_res1(::Val{:derivative}, f!, y, backend::AbstractADType, x, contexts...)
    return derivative(f!, y, backend, x, contexts...)
end
function get_res1(::Val{:gradient}, f, backend::AbstractADType, x, contexts...)
    return gradient(f, backend, x, contexts...)
end
function get_res1(::Val{:jacobian}, f, backend::AbstractADType, x, contexts...)
    return jacobian(f, backend, x, contexts...)
end
function get_res1(::Val{:jacobian}, f!, y, backend::AbstractADType, x, contexts...)
    return jacobian(f!, y, backend, x, contexts...)
end
function get_res1(::Val{:second_derivative}, f, backend::AbstractADType, x, contexts...)
    return derivative(f, backend, x, contexts...)
end
function get_res1(::Val{:hessian}, f, backend::AbstractADType, x, contexts...)
    return gradient(f, backend, x, contexts...)
end

function get_res2(::Val{:second_derivative}, f, backend::AbstractADType, x, contexts...)
    return second_derivative(f, backend, x, contexts...)
end
function get_res2(::Val{:hessian}, f, backend::AbstractADType, x, contexts...)
    return hessian(f, backend, x, contexts...)
end

function get_res1(::Val{:pushforward}, f, backend::AbstractADType, x, t, contexts...)
    return pushforward(f, backend, x, t, contexts...)
end
function get_res1(::Val{:pushforward}, f!, y, backend::AbstractADType, x, t, contexts...)
    return pushforward(f!, y, backend, x, t, contexts...)
end
function get_res1(::Val{:pullback}, f, backend::AbstractADType, x, t, contexts...)
    return pullback(f, backend, x, t, contexts...)
end
function get_res1(::Val{:pullback}, f!, y, backend::AbstractADType, x, t, contexts...)
    return pullback(f!, y, backend, x, t, contexts...)
end
function get_res1(::Val{:hvp}, f, backend::AbstractADType, x, t, contexts...)
    return gradient(f, backend, x, contexts...)
end

function get_res2(::Val{:hvp}, f, backend::AbstractADType, x, t, contexts...)
    return hvp(f, backend, x, t, contexts...)
end

"""
    compute_results(scen::Scenario, backend::AbstractADType)

Return a scenario identical to `scen` but where the first- and second-order results `res1` and `res2` have been computed with the given differentiation `backend`.

Useful for comparison of outputs between backends.
"""
function compute_results(
        scen::Scenario{op, pl_op, pl_fun}, backend::AbstractADType
    ) where {op, pl_op, pl_fun}
    (; f, y, x, t, contexts, prep_args, name) = deepcopy(scen)
    if pl_fun == :in
        if isnothing(t)
            new_res1 = get_res1(Val(op), f, y, backend, x, contexts...)
            new_res2 = get_res2(Val(op), f, y, backend, x, contexts...)
        else
            new_res1 = get_res1(Val(op), f, y, backend, x, t, contexts...)
            new_res2 = get_res2(Val(op), f, y, backend, x, t, contexts...)
        end
    else
        if isnothing(t)
            new_res1 = get_res1(Val(op), f, backend, x, contexts...)
            new_res2 = get_res2(Val(op), f, backend, x, contexts...)
        else
            new_res1 = get_res1(Val(op), f, backend, x, t, contexts...)
            new_res2 = get_res2(Val(op), f, backend, x, t, contexts...)
        end
    end
    new_scen = Scenario{op, pl_op, pl_fun}(;
        f, x, y, t, contexts, res1 = new_res1, res2 = new_res2, prep_args, name
    )
    return new_scen
end
