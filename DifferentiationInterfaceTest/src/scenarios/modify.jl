abstract type FunctionModifier end

"""
    zero(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the first- and second-order results which are set to zero.
"""
function Base.zero(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    return Scenario{op,pl_op,pl_fun}(
        scen.f;
        x=scen.x,
        y=scen.y,
        tang=scen.tang,
        contexts=scen.contexts,
        res1=myzero(scen.res1),
        res2=myzero(scen.res2),
    )
end

"""
    change_function(scen::Scenario, new_f)

Return a new `Scenario` identical to `scen` except for the function `f` which is changed to `new_f`.
"""
function change_function(scen::Scenario{op,pl_op,pl_fun}, new_f) where {op,pl_op,pl_fun}
    return Scenario{op,pl_op,pl_fun}(
        new_f;
        x=scen.x,
        y=scen.y,
        tang=scen.tang,
        contexts=scen.contexts,
        res1=scen.res1,
        res2=scen.res2,
    )
end

"""
    batchify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the tangents `tang` and associated results `res1` / `res2`, which are duplicated (batch mode).

Only works if `scen` is a `pushforward`, `pullback` or `hvp` scenario.
"""
function batchify(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, tang, contexts, res1, res2) = scen
    if op == :pushforward || op == :pullback
        new_tang = (only(tang), -only(tang))
        new_res1 = (only(res1), -only(res1))
        return Scenario{op,pl_op,pl_fun}(
            f; x, y, tang=new_tang, contexts, res1=new_res1, res2
        )
    elseif op == :hvp
        new_tang = (only(tang), -only(tang))
        new_res2 = (only(res2), -only(res2))
        return Scenario{op,pl_op,pl_fun}(
            f; x, y, tang=new_tang, contexts, res1, res2=new_res2
        )
    end
end

struct WritableClosure{pl_fun,F,X,Y} <: FunctionModifier
    f::F
    x_buffer::Vector{X}
    y_buffer::Vector{Y}
end

function WritableClosure{pl_fun}(
    f::F, x_buffer::Vector{X}, y_buffer::Vector{Y}
) where {pl_fun,F,X,Y}
    return WritableClosure{pl_fun,F,X,Y}(f, x_buffer, y_buffer)
end

Base.show(io::IO, f::WritableClosure) = print(io, "WritableClosure($(f.f))")

function (mc::WritableClosure{:out})(x)
    mc.x_buffer[1] = x
    mc.y_buffer[1] = mc.f(x)
    return copy(mc.y_buffer[1])
end

function (mc::WritableClosure{:in})(y, x)
    mc.x_buffer[1] = x
    mc.f(mc.y_buffer[1], mc.x_buffer[1])
    copyto!(y, mc.y_buffer[1])
    return nothing
end

"""
    closurify(scen::Scenario)

Return a new `Scenario` identical to `scen` except for the function `f` which is made to close over differentiable data.
"""
function closurify(scen::Scenario)
    (; f, x, y) = scen
    @assert isempty(scen.contexts)
    x_buffer = [zero(x)]
    y_buffer = [zero(y)]
    closure_f = WritableClosure{function_place(scen)}(f, x_buffer, y_buffer)
    return change_function(scen, closure_f)
end

struct MultiplyByConstant{pl_fun,F} <: FunctionModifier
    f::F
end

MultiplyByConstant{pl_fun}(f::F) where {pl_fun,F} = MultiplyByConstant{pl_fun,F}(f)

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

Return a new `Scenario` identical to `scen` except for the function `f`, which is made to accept an additional constant argument `a` by which the output is multiplied.
The output and result fields are updated accordingly.
"""
function constantify(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f,) = scen
    @assert isempty(scen.contexts)
    multiply_f = MultiplyByConstant{pl_fun}(f)
    a = 3.0
    return Scenario{op,pl_op,pl_fun}(
        multiply_f;
        x=scen.x,
        y=mymultiply(scen.y, a),
        tang=scen.tang,
        contexts=(Constant(a),),
        res1=mymultiply(scen.res1, a),
        res2=mymultiply(scen.res2, a),
    )
end

struct StoreInCache{pl_fun,F} <: FunctionModifier
    f::F
end

function StoreInCache{pl_fun}(f::F) where {pl_fun,F}
    return StoreInCache{pl_fun,F}(f)
end

Base.show(io::IO, f::StoreInCache) = print(io, "StoreInCache($(f.f))")

function (sc::StoreInCache{:out})(x, y_cache)
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

Return a new `Scenario` identical to `scen` except for the function `f`, which is made to accept an additional cache argument `a` to store the result before it is returned.
"""
function cachify(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f,) = scen
    @assert isempty(scen.contexts)
    cache_f = StoreInCache{pl_fun}(f)
    y_cache = if scen.y isa Number
        [myzero(scen.y)]
    else
        mysimilar(scen.y)
    end
    return Scenario{op,pl_op,pl_fun}(
        cache_f;
        x=scen.x,
        y=scen.y,
        tang=scen.tang,
        contexts=(Cache(y_cache),),
        res1=scen.res1,
        res2=scen.res2,
    )
end

function batchify(scens::AbstractVector{<:Scenario})
    batchifiable_scens = filter(s -> operator(s) in (:pushforward, :pullback, :hvp), scens)
    return batchify.(batchifiable_scens)
end

closurify(scens::AbstractVector{<:Scenario}) = closurify.(scens)
constantify(scens::AbstractVector{<:Scenario}) = constantify.(scens)
cachify(scens::AbstractVector{<:Scenario}) = cachify.(scens)
