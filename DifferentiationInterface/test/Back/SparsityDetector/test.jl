using Pkg
Pkg.add(["ForwardDiff", "ReverseDiff"])

using ADTypes: jacobian_sparsity, hessian_sparsity
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using LinearAlgebra
using SparseArrays
using StableRNGs
using Test

rng = StableRNG(63)

const Jc = sprand(rng, Bool, 10, 20, 0.3)
const Hc = sparse(Symmetric(sprand(rng, Bool, 20, 20, 0.3)))

f(x::AbstractVector) = Jc * x
f(x::AbstractMatrix) = reshape(f(vec(x)), (5, 2))

function f!(y, x)
    y .= f(x)
    return nothing
end

g(x::AbstractVector) = dot(x, Hc, x)
g(x::AbstractMatrix) = g(vec(x))

@testset verbose = true "$(typeof(backend))" for backend in
                                                 [AutoForwardDiff(), AutoReverseDiff()]
    @test_throws ArgumentError DenseSparsityDetector(backend; atol=1e-5, method=:random)
    @testset "$method" for method in (:iterative, :direct)
        detector = DenseSparsityDetector(backend; atol=1e-5, method)
        string(detector)
        for (x, y) in ((rand(20), zeros(10)), (rand(2, 10), zeros(5, 2)))
            @test Jc == jacobian_sparsity(f, x, detector)
            @test Jc == jacobian_sparsity(f!, copy(y), x, detector)
        end
        for x in (rand(20), rand(2, 10))
            @test Hc == hessian_sparsity(g, x, detector)
        end
    end
end
