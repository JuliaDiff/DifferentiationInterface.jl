myzero(x::Number) = zero(x)
myzero(x::AbstractArray) = zero(x)
myzero(x::Union{Tuple,NamedTuple}) = map(myzero, x)
myzero(::Nothing) = nothing

mysimilar(x::Number) = one(x)
mysimilar(x::AbstractArray) = similar(x)
mysimilar(x::Union{Tuple,NamedTuple}) = map(mysimilar, x)
mysimilar(x) = deepcopy(x)

myrandom(rng::AbstractRNG, x::Number) = randn(rng, typeof(x))
myrandom(rng::AbstractRNG, x::AbstractArray) = map(Base.Fix1(myrandom, rng), x)
myrandom(rng::AbstractRNG, x::Union{Tuple,NamedTuple}) = map(Base.Fix1(myrandom, rng), x)
myrandom(rng::AbstractRNG, x) = deepcopy(x)

myrandom(x) = myrandom(default_rng(), x)

mysize(x::Number) = size(x)
mysize(x::AbstractArray) = size(x)
mysize(x) = missing

mymultiply(x::Number, a::Number) = a * x
mymultiply(x::AbstractArray, a::Number) = a .* x
mymultiply(x::Union{Tuple,NamedTuple}, a::Number) = map(Base.Fix2(mymultiply, a), x)
mymultiply(::Nothing, a::Number) = nothing

mynnz(A::AbstractMatrix) = count(!iszero, A)
mynnz(A::AbstractSparseMatrix) = nnz(A)
