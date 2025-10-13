function DI.pick_batchsize(::AutoForwardDiff{nothing}, N::Integer)
    chunksize = pickchunksize(N)
    return DI.BatchSizeSettings{chunksize}(N)
end

function DI.pick_batchsize(::AutoForwardDiff{chunksize}, N::Integer) where {chunksize}
    return DI.BatchSizeSettings{chunksize}(N)
end

function DI.threshold_batchsize(
        backend::AutoForwardDiff{chunksize1}, chunksize2::Integer
    ) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoForwardDiff(; chunksize, tag = backend.tag)
end

choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{chunksize}, x) where {chunksize} = Chunk{chunksize}()

get_tag(f, backend::AutoForwardDiff, x) = backend.tag

function get_tag(f::F, backend::AutoForwardDiff{chunksize, Nothing}, x) where {F, chunksize}
    return Tag(f, eltype(x))
end

tag_type(::AutoForwardDiff{chunksize, T}) where {chunksize, T} = T
tag_type(f::F, backend::AutoForwardDiff, x) where {F} = typeof(get_tag(f, backend, x))

dual_type(config::DerivativeConfig) = eltype(config.duals)
dual_type(config::GradientConfig) = eltype(config.duals)
dual_type(config::JacobianConfig{T, V, N}) where {T, V, N} = Dual{T, V, N}
dual_type(config::HessianConfig) = dual_type(config.gradient_config)

function make_dual_similar(::Type{T}, x::Number, tx::NTuple{B}) where {T, B}
    return Dual{T}(x, tx...)
end

function make_dual_similar(::Type{T}, x, tx::NTuple{B}) where {T, B}
    return similar(x, Dual{T, eltype(x), B})
end

function make_dual(::Type{T}, x::Number, dx::Number) where {T}
    return Dual{T}(x, dx)
end

function make_dual(::Type{T}, x::Number, tx::NTuple{B}) where {T, B}
    return Dual{T}(x, tx...)
end

function make_dual(::Type{T}, x, tx::NTuple{B}) where {T, B}
    return Dual{T}.(x, tx...)
end

function make_dual!(::Type{T}, xdual, x, tx::NTuple{B}) where {T, B}
    return xdual .= Dual{T}.(x, tx...)
end

myvalue(::Type{T}, ydual::Number) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual) where {T} = myvalue.(T, ydual)
myvalue!(::Type{T}, y, ydual) where {T} = y .= myvalue.(T, ydual)

myderivative(::Type{T}, ydual::Number) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = myderivative.(T, ydual)
myderivative!(::Type{T}, dy, ydual) where {T} = dy .= myderivative.(T, ydual)

function mypartials(::Type{T}, ::Val{B}, ydual::Number) where {T, B}
    return ntuple(Val(B)) do b
        partials(T, ydual, b)
    end
end

function mypartials(::Type{T}, ::Val{B}, ydual) where {T, B}
    return ntuple(Val(B)) do b
        partials.(T, ydual, b)
    end
end

function mypartials!(::Type{T}, ty::NTuple{B}, ydual) where {T, B}
    for b in eachindex(ty)
        ty[b] .= partials.(T, ydual, b)
    end
    return ty
end

function _translate(
        ::Type{D}, c::Union{DI.GeneralizedConstant, DI.ConstantOrCache}
    ) where {D <: Dual}
    return DI.unwrap(c)
end
function _translate(::Type{D}, c::DI.Cache) where {D <: Dual}
    c0 = DI.unwrap(c)
    return DI.recursive_similar(c0, D)
end

function translate(::Type{D}, contexts::NTuple{C, DI.Context}) where {D <: Dual, C}
    new_contexts = map(contexts) do c
        _translate(D, c)
    end
    return new_contexts
end

function _translate_toprep(
        ::Type{D}, c::Union{DI.GeneralizedConstant, DI.ConstantOrCache}
    ) where {D <: Dual}
    return nothing
end
function _translate_toprep(::Type{D}, c::DI.Cache) where {D <: Dual}
    c0 = DI.unwrap(c)
    return DI.recursive_similar(c0, D)
end

function translate_toprep(::Type{D}, contexts::NTuple{C, DI.Context}) where {D <: Dual, C}
    new_contexts = map(contexts) do c
        _translate_toprep(D, c)
    end
    return new_contexts
end

function _translate_prepared(c::Union{DI.GeneralizedConstant, DI.ConstantOrCache}, _pc)
    return DI.unwrap(c)
end
_translate_prepared(_c::DI.Cache, pc) = pc

function translate_prepared(
        contexts::NTuple{C, DI.Context}, prep_contexts::NTuple{C, Any}
    ) where {C}
    new_contexts = map(contexts, prep_contexts) do c, pc
        _translate_prepared(c, pc)
    end
    return new_contexts
end
