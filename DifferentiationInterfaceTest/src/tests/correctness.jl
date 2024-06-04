## No overwrite

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        for n in fieldnames(typeof(scen))
            n in (:f, :ref, :first_order_ref) && continue
            @test getfield(new_scen, n) == getfield(scen, n)
        end
    end
end

## Pushforward

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dx) = new_scen = deepcopy(scen)
    dy_true = if ref_backend isa AbstractADType
        pushforward(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    for (k, extras) in enumerate([
        prepare_pushforward(f, ba, mycopy_random(x), mycopy_random(dx)),
        prepare_pushforward_same_point(f, ba, x, mycopy_random(dx)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            y1, dy1 = value_and_pushforward(f, ba, x, dx, extras)
            dy2 = pushforward(f, ba, x, dx, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PushforwardExtras
                end
                @testset "Primal value" begin
                    @test y1 ≈ y
                end
                @testset "Tangent value" begin
                    @test dy1 ≈ dy_true
                    @test dy2 ≈ dy_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dx) = new_scen = deepcopy(scen)
    dy_true = if ref_backend isa AbstractADType
        pushforward(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    for (k, extras) in enumerate([
        prepare_pushforward(f, ba, mycopy_random(x), mycopy_random(dx)),
        prepare_pushforward_same_point(f, ba, x, mycopy_random(dx)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            dy1_in = mysimilar(y)
            y1, dy1 = value_and_pushforward!(f, dy1_in, ba, x, dx, extras)

            dy2_in = mysimilar(y)
            dy2 = pushforward!(f, dy2_in, ba, x, dx, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PushforwardExtras
                end
                @testset "Primal value" begin
                    @test y1 ≈ y
                end
                @testset "Tangent value" begin
                    @test dy1_in ≈ dy_true
                    @test dy1 ≈ dy_true
                    @test dy2_in ≈ dy_true
                    @test dy2 ≈ dy_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{2,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dx) = new_scen = deepcopy(scen)
    f! = f
    dy_true = if ref_backend isa AbstractADType
        pushforward(f!, mysimilar(y), ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    for (k, extras) in enumerate([
        prepare_pushforward(f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(dx)),
        prepare_pushforward_same_point(f!, mysimilar(y), ba, x, mycopy_random(dx)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            y1_in = mysimilar(y)
            y1, dy1 = value_and_pushforward(f!, y1_in, ba, x, dx, extras)

            y2_in = mysimilar(y)
            dy2 = pushforward(f!, y2_in, ba, x, dx, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PushforwardExtras
                end
                @testset "Primal value" begin
                    @test y1_in ≈ y
                    @test y1 ≈ y
                end
                @testset "Tangent value" begin
                    @test dy1 ≈ dy_true
                    @test dy2 ≈ dy_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dx) = new_scen = deepcopy(scen)
    f! = f
    dy_true = if ref_backend isa AbstractADType
        pushforward(f!, mysimilar(y), ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    for (k, extras) in enumerate([
        prepare_pushforward(f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(dx)),
        prepare_pushforward_same_point(f!, mysimilar(y), ba, x, mycopy_random(dx)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            y1_in, dy1_in = mysimilar(y), mysimilar(y)
            y1, dy1 = value_and_pushforward!(f!, y1_in, dy1_in, ba, x, dx, extras)

            y2_in, dy2_in = mysimilar(y), mysimilar(y)
            dy2 = pushforward!(f!, y2_in, dy2_in, ba, x, dx, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PushforwardExtras
                end
                @testset "Primal value" begin
                    @test y1_in ≈ y
                    @test y1 ≈ y
                end
                @testset "Tangent value" begin
                    @test dy1_in ≈ dy_true
                    @test dy1 ≈ dy_true
                    @test dy2_in ≈ dy_true
                    @test dy2 ≈ dy_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Pullback

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dy) = new_scen = deepcopy(scen)
    dx_true = if ref_backend isa AbstractADType
        pullback(f, ref_backend, x, dy)
    else
        new_scen.ref(x, dy)
    end

    for (k, extras) in enumerate([
        prepare_pullback(f, ba, mycopy_random(x), mycopy_random(dy)),
        prepare_pullback_same_point(f, ba, x, mycopy_random(dy)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            y1, dx1 = value_and_pullback(f, ba, x, dy, extras)

            dx2 = pullback(f, ba, x, dy, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PullbackExtras
                end
                @testset "Primal value" begin
                    @test y1 ≈ y
                end
                @testset "Cotangent value" begin
                    @test dx1 ≈ dx_true
                    @test dx2 ≈ dx_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dy) = new_scen = deepcopy(scen)
    dx_true = if ref_backend isa AbstractADType
        pullback(f, ref_backend, x, dy)
    else
        new_scen.ref(x, dy)
    end

    for (k, extras) in enumerate([
        prepare_pullback(f, ba, mycopy_random(x), mycopy_random(dy)),
        prepare_pullback_same_point(f, ba, x, mycopy_random(dy)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            dx1_in = mysimilar(x)
            y1, dx1 = value_and_pullback!(f, dx1_in, ba, x, dy, extras)

            dx2_in = mysimilar(x)
            dx2 = pullback!(f, dx2_in, ba, x, dy, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PullbackExtras
                end
                @testset "Primal value" begin
                    @test y1 ≈ y
                end
                @testset "Cotangent value" begin
                    @test dx1_in ≈ dx_true
                    @test dx1 ≈ dx_true
                    @test dx2_in ≈ dx_true
                    @test dx2 ≈ dx_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{2,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dy) = new_scen = deepcopy(scen)
    f! = f
    dx_true = if ref_backend isa AbstractADType
        pullback(f!, mysimilar(y), ref_backend, x, dy)
    else
        new_scen.ref(x, dy)
    end

    for (k, extras) in enumerate([
        prepare_pullback(f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(dy)),
        prepare_pullback_same_point(f!, mysimilar(y), ba, x, mycopy_random(dy)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            y1_in = mysimilar(y)
            y1, dx1 = value_and_pullback(f!, y1_in, ba, x, dy, extras)

            y2_in = mysimilar(y)
            dx2 = pullback(f!, y2_in, ba, x, dy, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PullbackExtras
                end
                @testset "Primal value" begin
                    @test y1_in ≈ y
                    @test y1 ≈ y
                end
                @testset "Cotangent value" begin
                    @test dx1 ≈ dx_true
                    @test dx2 ≈ dx_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y, dy) = new_scen = deepcopy(scen)
    f! = f
    dx_true = if ref_backend isa AbstractADType
        pullback(f!, mysimilar(y), ref_backend, x, dy)
    else
        new_scen.ref(x, dy)
    end

    for (k, extras) in enumerate([
        prepare_pullback(f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(dy)),
        prepare_pullback_same_point(f!, mysimilar(y), ba, x, mycopy_random(dy)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            y1_in, dx1_in = mysimilar(y), mysimilar(x)
            y1, dx1 = value_and_pullback!(f!, y1_in, dx1_in, ba, x, dy, extras)

            y2_in, dx2_in = mysimilar(y), mysimilar(x)
            dx2 = pullback!(f!, y2_in, dx2_in, ba, x, dy, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa PullbackExtras
                end
                @testset "Primal value" begin
                    @test y1_in ≈ y
                    @test y1 ≈ y
                end
                @testset "Cotangent value" begin
                    @test dx1_in ≈ dx_true
                    @test dx1 ≈ dx_true
                    @test dx2_in ≈ dx_true
                    @test dx2 ≈ dx_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Derivative

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_derivative(f, ba, mycopy_random(x))
    der_true = if ref_backend isa AbstractADType
        derivative(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1, der1 = value_and_derivative(f, ba, x, extras)

    der2 = derivative(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa DerivativeExtras
        end
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1 ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_derivative(f, ba, mycopy_random(x))
    der_true = if ref_backend isa AbstractADType
        derivative(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    der1_in = mysimilar(y)
    y1, der1 = value_and_derivative!(f, der1_in, ba, x, extras)

    der2_in = mysimilar(y)
    der2 = derivative!(f, der2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa DerivativeExtras
        end
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1_in ≈ der_true
            @test der1 ≈ der_true
            @test der2_in ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{2,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, mysimilar(y), ba, mycopy_random(x))
    der_true = if ref_backend isa AbstractADType
        derivative(f!, mysimilar(y), ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1_in = mysimilar(y)
    y1, der1 = value_and_derivative(f!, y1_in, ba, x, extras)

    y2_in = mysimilar(y)
    der2 = derivative(f!, y2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa DerivativeExtras
        end
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1 ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, mysimilar(y), ba, mycopy_random(x))
    der_true = if ref_backend isa AbstractADType
        derivative(f!, mysimilar(y), ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1_in, der1_in = mysimilar(y), mysimilar(y)
    y1, der1 = value_and_derivative!(f!, y1_in, der1_in, ba, x, extras)

    y2_in, der2_in = mysimilar(y), mysimilar(y)
    der2 = derivative!(f!, y2_in, der2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa DerivativeExtras
        end
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1_in ≈ der_true
            @test der1 ≈ der_true
            @test der2_in ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Gradient

function test_correctness(
    ba::AbstractADType,
    scen::GradientScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_gradient(f, ba, mycopy_random(x))
    grad_true = if ref_backend isa AbstractADType
        gradient(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1, grad1 = value_and_gradient(f, ba, x, extras)

    grad2 = gradient(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa GradientExtras
        end
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Gradient value" begin
            @test grad1 ≈ grad_true
            @test grad2 ≈ grad_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::GradientScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_gradient(f, ba, mycopy_random(x))
    grad_true = if ref_backend isa AbstractADType
        gradient(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    grad1_in = mysimilar(x)
    y1, grad1 = value_and_gradient!(f, grad1_in, ba, x, extras)

    grad2_in = mysimilar(x)
    grad2 = gradient!(f, grad2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa GradientExtras
        end
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Gradient value" begin
            @test grad1_in ≈ grad_true
            @test grad1 ≈ grad_true
            @test grad2_in ≈ grad_true
            @test grad2 ≈ grad_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Jacobian

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, mycopy_random(x))
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1, jac1 = value_and_jacobian(f, ba, x, extras)

    jac2 = jacobian(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa JacobianExtras
        end
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1 ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, mycopy_random(x))
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    jac1_in = mysimilar(jac_true)
    y1, jac1 = value_and_jacobian!(f, jac1_in, ba, x, extras)

    jac2_in = mysimilar(jac_true)
    jac2 = jacobian!(f, jac2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa JacobianExtras
        end
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1_in ≈ jac_true
            @test jac1 ≈ jac_true
            @test jac2_in ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{2,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, mycopy_random(x))
    jac_true = if ref_backend isa AbstractADType
        jacobian(f!, mysimilar(y), ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1_in = mysimilar(y)
    y1, jac1 = value_and_jacobian(f!, y1_in, ba, x, extras)

    y2_in = mysimilar(y)
    jac2 = jacobian(f!, y2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa JacobianExtras
        end
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1 ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, mycopy_random(x))
    jac_true = if ref_backend isa AbstractADType
        jacobian(f!, mysimilar(y), ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1_in, jac1_in = mysimilar(y), mysimilar(jac_true)
    y1, jac1 = value_and_jacobian!(f!, y1_in, jac1_in, ba, x, extras)

    y2_in, jac2_in = mysimilar(y), mysimilar(jac_true)
    jac2 = jacobian!(f!, y2_in, jac2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa JacobianExtras
        end
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1_in ≈ jac_true
            @test jac1 ≈ jac_true
            @test jac2_in ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Second derivative

function test_correctness(
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, mycopy_random(x))
    der1_true = if ref_backend isa SecondOrder
        derivative(f, inner(ref_backend), x)
    elseif ref_backend isa AbstractADType
        derivative(f, ref_backend, x)
    else
        new_scen.first_order_ref(x)
    end
    der2_true = if ref_backend isa AbstractADType
        second_derivative(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    der21 = second_derivative(f, ba, x, extras)
    y2, der12, der22 = value_derivative_and_second_derivative(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa SecondDerivativeExtras
        end
        @testset "Primal value" begin
            @test y2 ≈ y
        end
        @testset "First derivative value" begin
            @test der12 ≈ der1_true
        end
        @testset "Second derivative value" begin
            @test der21 ≈ der2_true
            @test der22 ≈ der2_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, mycopy_random(x))
    der1_true = if ref_backend isa SecondOrder
        derivative(f, inner(ref_backend), x)
    elseif ref_backend isa AbstractADType
        derivative(f, ref_backend, x)
    else
        new_scen.first_order_ref(x)
    end
    der2_true = if ref_backend isa AbstractADType
        second_derivative(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    der21_in = mysimilar(y)
    der21 = second_derivative!(f, der21_in, ba, x, extras)

    der12_in, der22_in = mysimilar(y), mysimilar(y)
    y2, der12, der22 = value_derivative_and_second_derivative!(
        f, der12_in, der22_in, ba, x, extras
    )

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa SecondDerivativeExtras
        end
        @testset "Primal value" begin
            @test y2 ≈ y
        end
        @testset "Derivative value" begin
            @test der12_in ≈ der1_true
            @test der12 ≈ der1_true
        end
        @testset "Second derivative value" begin
            @test der21_in ≈ der2_true
            @test der22_in ≈ der2_true
            @test der21 ≈ der2_true
            @test der22 ≈ der2_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian-vector product

function test_correctness(
    ba::AbstractADType,
    scen::HVPScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, dx) = new_scen = deepcopy(scen)
    p_true = if ref_backend isa AbstractADType
        hvp(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    for (k, extras) in enumerate([
        prepare_hvp(f, ba, mycopy_random(x), mycopy_random(dx)),
        prepare_hvp_same_point(f, ba, x, mycopy_random(dx)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            p1 = hvp(f, ba, x, dx, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa HVPExtras
                end
                @testset "HVP value" begin
                    @test p1 ≈ p_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::HVPScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, dx) = new_scen = deepcopy(scen)
    p_true = if ref_backend isa AbstractADType
        hvp(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    for (k, extras) in enumerate([
        prepare_hvp(f, ba, mycopy_random(x), mycopy_random(dx)),
        prepare_hvp_same_point(f, ba, x, mycopy_random(dx)),
    ])
        testset_name = k == 1 ? "Different point" : "Same point"
        @testset "$testset_name" begin
            p1_in = mysimilar(x)
            p1 = hvp!(f, p1_in, ba, x, dx, extras)

            let (≈)(x, y) = isapprox(x, y; atol, rtol)
                @testset "Extras type" begin
                    @test extras isa HVPExtras
                end
                @testset "HVP value" begin
                    @test p1_in ≈ p_true
                    @test p1 ≈ p_true
                end
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian

function test_correctness(
    ba::AbstractADType,
    scen::HessianScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, mycopy_random(x))
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa HessianExtras
        end
        @testset "Hessian value" begin
            @test hess1 ≈ hess_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::HessianScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, mycopy_random(x))
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1_in = mysimilar(hess_true)
    hess1 = hessian!(f, hess1_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Extras type" begin
            @test extras isa HessianExtras
        end
        @testset "Hessian value" begin
            @test hess1_in ≈ hess_true
            @test hess1 ≈ hess_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end
