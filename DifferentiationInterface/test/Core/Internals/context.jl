using DifferentiationInterface
using DifferentiationInterface:
    AnyConstant, GeneralizedConstant, Rewrap, adapt_eltype, fix_tail, maker, unwrap
using Test

f1(x) = x
g1 = @inferred fix_tail(f1)
@test @inferred g1(4) == 4

f2(x, a, b) = a * x + b
g2 = @inferred fix_tail(f2, 2, 3)
@test @inferred g2(4) == 2 * 4 + 3

contexts = ()
r = @inferred Rewrap()
@test r() == ()

contexts = (Constant(1.0), Cache([2.0]))
r = @inferred Rewrap(contexts...)
@test (@inferred r(3.0, [4.0])) == (Constant(3.0), Cache([4.0]))
@test (@inferred r(3, [4.0f0])) isa Tuple{Constant{Int}, Cache{Vector{Float32}}}

contexts = (PrepTimeConstant(1.0), Cache([2.0]))
r = @inferred Rewrap(contexts...)
@test (@inferred r(3.0, [4.0])) == (PrepTimeConstant(3.0), Cache([4.0]))
@test (@inferred r(3, [4.0f0])) isa Tuple{PrepTimeConstant{Int}, Cache{Vector{Float32}}}

# PrepTimeConstant behaves like Constant in the type hierarchy
@test PrepTimeConstant <: GeneralizedConstant
@test PrepTimeConstant <: AnyConstant
@test Constant <: AnyConstant
@test !(Cache <: AnyConstant)
@test !(ConstantOrCache <: AnyConstant)
@test unwrap(PrepTimeConstant(2.0)) == 2.0
@test PrepTimeConstant(2.0) == PrepTimeConstant(2.0)

# `maker` and `adapt_eltype` are the two hooks every context type must provide
@test maker(PrepTimeConstant(1.0))(2.0) === PrepTimeConstant(2.0)
@test adapt_eltype(PrepTimeConstant(1.0), Float32) === PrepTimeConstant(1.0)
@test adapt_eltype(PrepTimeConstant([1.0]), Float32) isa PrepTimeConstant{Vector{Float64}}
