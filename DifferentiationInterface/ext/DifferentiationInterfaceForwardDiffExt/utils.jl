function DI.BatchSizeSettings(::AutoForwardDiff{nothing}, N::Integer)
    chunksize = ForwardDiff.pickchunksize(N)
    return DI.BatchSizeSettings{chunksize}(N)
end

function DI.BatchSizeSettings(::AutoForwardDiff{chunksize}, N::Integer) where {chunksize}
    return DI.BatchSizeSettings{chunksize}(N)
end

function DI.threshold_batchsize(
    backend::AutoForwardDiff{chunksize1}, chunksize2::Integer
) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoForwardDiff(; chunksize, tag=backend.tag)
end

choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{chunksize}, x) where {chunksize} = Chunk{chunksize}()

get_tag(f, backend::AutoForwardDiff, x) = backend.tag

function get_tag(f::F, ::AutoForwardDiff{chunksize,Nothing}, x) where {F,chunksize}
    return Tag(f, eltype(x))
end

tag_type(::AutoForwardDiff{chunksize,T}) where {chunksize,T} = T
tag_type(f::F, backend::AutoForwardDiff, x) where {F} = typeof(get_tag(f, backend, x))

function make_dual_similar(::Type{T}, x::Number, tx::NTuple{B}) where {T,B}
    return Dual{T}(x, tx...)
end

function make_dual_similar(::Type{T}, x, tx::NTuple{B}) where {T,B}
    return similar(x, Dual{T,eltype(x),B})
end

function make_dual(::Type{T}, x::Number, dx::Number) where {T}
    return Dual{T}(x, dx)
end

function make_dual(::Type{T}, x::Number, tx::NTuple{B}) where {T,B}
    return Dual{T}(x, tx...)
end

function make_dual(::Type{T}, x, tx::NTuple{B}) where {T,B}
    return Dual{T}.(x, tx...)
end

function make_dual!(::Type{T}, xdual, x, tx::NTuple{B}) where {T,B}
    return xdual .= Dual{T}.(x, tx...)
end

myvalue(::Type{T}, ydual::Number) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual) where {T} = myvalue.(T, ydual)
myvalue!(::Type{T}, y, ydual) where {T} = y .= myvalue.(T, ydual)

myderivative(::Type{T}, ydual::Number) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = myderivative.(T, ydual)
myderivative!(::Type{T}, dy, ydual) where {T} = dy .= myderivative.(T, ydual)

function mypartials(::Type{T}, ::Val{B}, ydual::Number) where {T,B}
    return ntuple(Val(B)) do b
        partials(T, ydual, b)
    end
end

function mypartials(::Type{T}, ::Val{B}, ydual) where {T,B}
    return ntuple(Val(B)) do b
        partials.(T, ydual, b)
    end
end

function mypartials!(::Type{T}, ty::NTuple{B}, ydual) where {T,B}
    for b in eachindex(ty)
        ty[b] .= partials.(T, ydual, b)
    end
    return ty
end

# store preparation result with the right input eltype
struct PrepContext{T<:DI.Prep} <: DI.Context
    data::T
end

function _translate(::Type{T}, ::Val{B}, c::DI.ConstantOrFunctionOrBackend) where {T,B}
    return DI.unwrap(c)
end
_translate(::Type{T}, ::Val{B}, c::PrepContext) where {T,B} = DI.unwrap(c)

function _translate(::Type{T}, ::Val{B}, c::DI.Cache) where {T,B}
    c0 = DI.unwrap(c)
    return make_dual(T, c0, ntuple(_ -> similar(c0), Val(B)))  # TODO: optimize
end

function translate(::Type{T}, ::Val{B}, contexts::Vararg{DI.Context,C}) where {T,B,C}
    new_contexts = map(contexts) do c
        _translate(T, Val(B), c)
    end
    return new_contexts
end
