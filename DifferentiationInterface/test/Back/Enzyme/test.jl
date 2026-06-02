using DifferentiationInterface
import Enzyme
using Test

backend = SecondOrder(
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward)),
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
)

struct StoreInCache{F}
    f::F
end

function (sc::StoreInCache)(x, y_cache)
    y_cache[1] = sc.f(x)
    return y_cache[1]
end

f(x::AbstractArray) = sum(vec(x .^ 4) .* transpose(vec(x .^ 6)))

x = [1.0;;]
dx = [3.0;;]
c = similar(x)
@test_nowarn hvp(StoreInCache(f), backend, x, (dx,), Cache(x))
