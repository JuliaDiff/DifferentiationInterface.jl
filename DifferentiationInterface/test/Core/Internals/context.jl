using DifferentiationInterface
using DifferentiationInterface: Rewrap, fix_tail
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
@test (@inferred r(3, [4.0f0])) isa Tuple{Constant{Int},Cache{Vector{Float32}}}
