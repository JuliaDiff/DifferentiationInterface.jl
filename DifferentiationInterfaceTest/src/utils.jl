mysimilar(x::AbstractArray) = similar(x)
mysimilar(x::Union{Tuple, NamedTuple}) = map(mysimilar, x)

mysize(x::Number) = size(x)
mysize(x::AbstractArray) = size(x)
mysize(x) = missing

mymultiply(x::Number, a::Number) = a * x
mymultiply(x::AbstractArray, a::Number) = a .* x
mymultiply(x::Union{Tuple, NamedTuple}, a::Number) = map(Base.Fix2(mymultiply, a), x)
mymultiply(::Nothing, a::Number) = nothing

mynnz(A::AbstractMatrix) = count(!iszero, A)
mynnz(A::AbstractSparseMatrix) = nnz(A)
