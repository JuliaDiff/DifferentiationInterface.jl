using DifferentiationInterface
using DifferentiationInterface: AutoZeroReverse, AutoZeroForward
using Test

backend = AutoZeroForward()
other_backend = AutoZeroReverse()
f(x, c) = x + c
f!(y, x) = y .= x
x = 1.0
y = zeros(2)
c = 2.0

@testset "Out of place, no tangents" begin
    prep = prepare_derivative(f, backend, x, Constant(c); strict=Val(true))
    prep_chill = prepare_derivative(f, backend, x, Constant(c); strict=Val(false))

    @test_throws MethodError derivative(nothing, prep_chill, backend, x, Constant(c))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f: ❌
        - prep: typeof(f)
        - exec: Nothing
      - backend: ✅
      - x: ✅
      - contexts: ✅
    """ derivative(nothing, prep, backend, x, Constant(c))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f: ✅
      - backend: ❌
        - prep: AutoZeroForward
        - exec: AutoZeroReverse
      - x: ✅
      - contexts: ✅
    """ derivative(f, prep, other_backend, x, Constant(c))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f: ✅
      - backend: ✅
      - x: ❌
        - prep: Float64
        - exec: Int64
      - contexts: ✅
    """ derivative(f, prep, backend, 1, Constant(c))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f: ✅
      - backend: ✅
      - x: ✅
      - contexts: ❌
        - prep: Tuple{Constant{Float64}}
        - exec: Tuple{Constant{Int64}}
    """ derivative(f, prep, backend, x, Constant(2))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f: ✅
      - backend: ✅
      - x: ✅
      - contexts: ❌
        - prep: Tuple{Constant{Float64}}
        - exec: Tuple{Constant{Int64}, Constant{Int64}}
    """ derivative(f, prep, backend, x, Constant(2), Constant(3))
end

@testset "In place, no tangents" begin
    prep = prepare_derivative(f!, y, backend, x; strict=Val(true))
    prep_chill = prepare_derivative(f!, y, backend, x; strict=Val(false))

    @test_throws MethodError derivative(nothing, y, prep_chill, backend, x, Constant(c))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f!: ❌
        - prep: typeof(f!)
        - exec: Nothing
      - y: ✅
      - backend: ✅
      - x: ✅
      - contexts: ✅
    """ derivative(nothing, y, prep, backend, x)
end

@testset "Out of place, with tangents" begin
    prep = prepare_pushforward(f, backend, x, (x,), Constant(c); strict=Val(true))
    prep_chill = prepare_pushforward(f, backend, x, (x,), Constant(c); strict=Val(false))

    @test_throws MethodError pushforward(nothing, prep_chill, backend, x, (x,))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f: ❌
        - prep: typeof(f)
        - exec: Nothing
      - backend: ✅
      - x: ✅
      - tang: ✅
      - contexts: ✅
    """ pushforward(nothing, prep, backend, x, (x,), Constant(c))
end

@testset "In place, with tangents" begin
    prep = prepare_pushforward(f!, y, backend, x, (x,); strict=Val(true))
    prep_chill = prepare_pushforward(
        f!, y, backend, x, (x,), Constant(c); strict=Val(false)
    )

    @test_throws MethodError pushforward(nothing, y, prep_chill, backend, x, (x,))

    @test_throws """
    PreparationMismatchError (inconsistent types between preparation and execution):
      - f!: ❌
        - prep: typeof(f!)
        - exec: Nothing
      - y: ✅
      - backend: ✅
      - x: ✅
      - tang: ✅
      - contexts: ✅
    """ pushforward(nothing, y, prep, backend, x, (x,))
end
