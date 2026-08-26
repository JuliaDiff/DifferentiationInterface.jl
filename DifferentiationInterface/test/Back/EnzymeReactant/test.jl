using DifferentiationInterface:
    AutoEnzyme,
    Constant,
    jacobian,
    jacobian!,
    prepare_jacobian,
    value_and_jacobian
using Enzyme: Enzyme
using Reactant: Reactant, @jit
using Test

f(x, p) = x .^ 2 .- p
f!(y, x, p) = y .= f(x, p)

function oop_jacobian(x, p)
    backend = AutoEnzyme()
    prep = prepare_jacobian(f, backend, x, Constant(p))
    return jacobian(f, prep, backend, x, Constant(p))
end

function oop_value_and_jacobian(x, p)
    backend = AutoEnzyme()
    prep = prepare_jacobian(f, backend, x, Constant(p))
    return value_and_jacobian(f, prep, backend, x, Constant(p))
end

function iip_jacobian(x, p)
    backend = AutoEnzyme()
    y = zero(x)
    prep = prepare_jacobian(f!, y, backend, x, Constant(p))
    return jacobian(f!, y, prep, backend, x, Constant(p))
end

function iip_value_and_jacobian(x, p)
    backend = AutoEnzyme()
    y = zero(x)
    prep = prepare_jacobian(f!, y, backend, x, Constant(p))
    return value_and_jacobian(f!, y, prep, backend, x, Constant(p))
end

function oop_jacobian!(x, p)
    backend = AutoEnzyme()
    jac = similar(x, length(x), length(x))
    prep = prepare_jacobian(f, backend, x, Constant(p))
    return jacobian!(f, jac, prep, backend, x, Constant(p))
end

function iip_jacobian!(x, p)
    backend = AutoEnzyme()
    y = zero(x)
    jac = similar(x, length(x), length(x))
    prep = prepare_jacobian(f!, y, backend, x, Constant(p))
    return jacobian!(f!, y, jac, prep, backend, x, Constant(p))
end

@testset "AutoEnzyme Jacobian inside Reactant" begin
    x = Reactant.to_rarray(Float32[1, 2])
    p = Reactant.to_rarray(Float32[3, 4])
    expected_jacobian = Float32[2 0; 0 4]

    @test @jit(oop_jacobian(x, p)) ≈ expected_jacobian
    value, jac = @jit oop_value_and_jacobian(x, p)
    @test value ≈ Float32[-2, 0]
    @test jac ≈ expected_jacobian
    @test @jit(iip_jacobian(x, p)) ≈ expected_jacobian
    value, jac = @jit iip_value_and_jacobian(x, p)
    @test value ≈ Float32[-2, 0]
    @test jac ≈ expected_jacobian
    @test @jit(oop_jacobian!(x, p)) ≈ expected_jacobian
    @test @jit(iip_jacobian!(x, p)) ≈ expected_jacobian
end
