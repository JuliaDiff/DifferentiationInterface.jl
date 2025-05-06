const PME = PreparationMismatchError

for op in ALL_OPS
    op! = Symbol(op, "!")
    val_prefix = if op == :second_derivative
        "value_derivative_and_"
    elseif op == :hessian
        "value_gradient_and_"
    elseif op == :hvp
        "gradient_and_"
    else
        "value_and_"
    end
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)
    prep_op! = Symbol("prepare!_", op)
    prep_op_same = Symbol("prepare_", op, "_same_point")

    P = if op == :derivative
        DerivativePrep
    elseif op == :gradient
        GradientPrep
    elseif op == :hessian
        HessianPrep
    elseif op == :hvp
        HVPPrep
    elseif op == :jacobian
        JacobianPrep
    elseif op == :pullback
        PullbackPrep
    elseif op == :pushforward
        PushforwardPrep
    elseif op == :second_derivative
        SecondDerivativePrep
    end

    S1out = Scenario{op,:out,:out}
    S1in = Scenario{op,:in,:out}
    S2out = Scenario{op,:out,:in}
    S2in = Scenario{op,:in,:in}

    if op in [:derivative, :gradient, :jacobian]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand = myrandom(x)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    contextsrand...,
                )
                [(), (prep,), (prepstrict,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_out1_val, res1_out1_val = $val_and_op(
                    f, preptup_val..., ba, x, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op(
                    f, preptup_val..., ba, x, contexts...
                )
                res1_out1_noval = $op(f, preptup_noval..., ba, x, contexts...)
                res1_out2_noval = $op(f, preptup_noval..., ba, x, contexts...)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
                if sparsity && $op == jacobian
                    @test mynnz(res1_out1_val) == mynnz(scen.res1)
                    @test mynnz(res1_out2_val) == mynnz(scen.res1)
                    @test mynnz(res1_out1_noval) == mynnz(scen.res1)
                    @test mynnz(res1_out2_noval) == mynnz(scen.res1)
                end
            end
            @test_throws PME $val_and_op(nothing, prepstrict, ba, x, contexts...)
            @test_throws PME $op(nothing, prepstrict, ba, x, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand = myrandom(x)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    contextsrand...,
                )
                [(), (prep,), (prepstrict,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                res1_in1_val = mysimilar(res1)
                res1_in2_val = mysimilar(res1)
                res1_in1_noval = mysimilar(res1)
                res1_in2_noval = mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, res1_in1_val, preptup_val..., ba, x, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, res1_in2_val, preptup_val..., ba, x, contexts...
                )
                res1_out1_noval = $op!(
                    f, res1_in1_noval, preptup_noval..., ba, x, contexts...
                )
                res1_out2_noval = $op!(
                    f, res1_in2_noval, preptup_noval..., ba, x, contexts...
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val === res1_out1_val
                    @test res1_in2_val === res1_out2_val
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_in1_noval === res1_out1_noval
                    @test res1_in2_noval === res1_out2_noval
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
                if sparsity && $op == jacobian
                    @test mynnz(res1_out1_val) == mynnz(scen.res1)
                    @test mynnz(res1_out2_val) == mynnz(scen.res1)
                    @test mynnz(res1_out1_noval) == mynnz(scen.res1)
                    @test mynnz(res1_out2_noval) == mynnz(scen.res1)
                end
            end
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), prepstrict, ba, x, contexts...
            )
            @test_throws PME $op!(nothing, mysimilar(res1), prepstrict, ba, x, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

        op == :gradient && continue

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2out;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, yrand = myrandom(x), myrandom(y)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, copy(yrand), ba, xrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    copy(yrand),
                    $prep_op(
                        new_smaller.f,
                        copy(new_smaller.y),
                        ba,
                        new_smaller.x,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    contextsrand...,
                )
                [(), (prep,), (prepstrict,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_in1_val = mysimilar(y)
                y_in2_val = mysimilar(y)
                y_in1_noval = mysimilar(y)
                y_in2_noval = mysimilar(y)
                y_out1_val, res1_out1_val = $val_and_op(
                    f, y_in1_val, preptup_val..., ba, x, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op(
                    f, y_in2_val, preptup_val..., ba, x, contexts...
                )
                res1_out1_noval = $op(f, y_in1_noval, preptup_noval..., ba, x, contexts...)
                res1_out2_noval = $op(f, y_in2_noval, preptup_noval..., ba, x, contexts...)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_in1_val === y_out1_val
                    @test y_in2_val === y_out2_val
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
                if sparsity && $op == jacobian
                    @test mynnz(res1_out1_val) == mynnz(scen.res1)
                    @test mynnz(res1_out2_val) == mynnz(scen.res1)
                    @test mynnz(res1_out1_noval) == mynnz(scen.res1)
                    @test mynnz(res1_out2_noval) == mynnz(scen.res1)
                end
            end
            @test_throws PME $val_and_op(
                nothing, mysimilar(y), prepstrict, ba, x, contexts...
            )
            @test_throws PME $op(nothing, mysimilar(y), prepstrict, ba, x, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2in;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, yrand = myrandom(x), myrandom(y)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, copy(yrand), ba, xrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    copy(yrand),
                    $prep_op(
                        new_smaller.f,
                        copy(new_smaller.y),
                        ba,
                        new_smaller.x,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    contextsrand...,
                )
                [(), (prep,), (prepstrict,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_in1_val, res1_in1_val = mysimilar(y), mysimilar(res1)
                y_in2_val, res1_in2_val = mysimilar(y), mysimilar(res1)
                y_in1_noval, res1_in1_noval = mysimilar(y), mysimilar(res1)
                y_in2_noval, res1_in2_noval = mysimilar(y), mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, y_in1_val, res1_in1_val, preptup_val..., ba, x, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, y_in2_val, res1_in2_val, preptup_val..., ba, x, contexts...
                )
                res1_out1_noval = $op!(
                    f, y_in1_noval, res1_in1_noval, preptup_noval..., ba, x, contexts...
                )
                res1_out2_noval = $op!(
                    f, y_in2_noval, res1_in2_noval, preptup_noval..., ba, x, contexts...
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_in1_val === y_out1_val
                    @test y_in2_val === y_out2_val
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val === res1_out1_val
                    @test res1_in2_val === res1_out2_val
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_in1_noval === res1_out1_noval
                    @test res1_in2_noval === res1_out2_noval
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
                if sparsity && $op == jacobian
                    @test mynnz(res1_out1_val) == mynnz(scen.res1)
                    @test mynnz(res1_out2_val) == mynnz(scen.res1)
                    @test mynnz(res1_out1_noval) == mynnz(scen.res1)
                    @test mynnz(res1_out2_noval) == mynnz(scen.res1)
                end
            end
            @test_throws PME $val_and_op!(
                nothing, mysimilar(y), mysimilar(res1), prepstrict, ba, x, contexts...
            )
            @test_throws PME $op!(
                nothing, mysimilar(y), mysimilar(res1), prepstrict, ba, x, contexts...
            )
            scenario_intact && @test new_scen == scen
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, res1, res2, contexts, smaller) = new_scen = deepcopy(scen)
            xrand = myrandom(x)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    contextsrand...,
                )
                [(), (prep,), (prepstrict,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_out1_val, res1_out1_val, res2_out1_val = $val_and_op(
                    f, preptup_val..., ba, x, contexts...
                )
                y_out2_val, res1_out2_val, res2_out2_val = $val_and_op(
                    f, preptup_val..., ba, x, contexts...
                )
                res2_out1_noval = $op(f, preptup_noval..., ba, x, contexts...)
                res2_out2_noval = $op(f, preptup_noval..., ba, x, contexts...)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res2_out1_val ≈ scen.res2
                    @test res2_out2_val ≈ scen.res2
                    @test res2_out1_noval ≈ scen.res2
                    @test res2_out2_noval ≈ scen.res2
                end
                if sparsity && $op == hessian
                    @test mynnz(res2_out1_val) == mynnz(scen.res2)
                    @test mynnz(res2_out2_val) == mynnz(scen.res2)
                    @test mynnz(res2_out1_noval) == mynnz(scen.res2)
                    @test mynnz(res2_out2_noval) == mynnz(scen.res2)
                end
            end
            @test_throws PME $val_and_op(nothing, prepstrict, ba, x, contexts...)
            @test_throws PME $op(nothing, prepstrict, ba, x, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, res1, res2, contexts, smaller) = new_scen = deepcopy(scen)
            xrand = myrandom(x)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    contextsrand...,
                )
                [(), (prep,), (prepstrict,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                res1_in1_val, res2_in1_val = mysimilar(res1), mysimilar(res2)
                res1_in2_val, res2_in2_val = mysimilar(res1), mysimilar(res2)
                res2_in1_noval = mysimilar(res2)
                res2_in2_noval = mysimilar(res2)
                y_out1_val, res1_out1_val, res2_out1_val = $val_and_op!(
                    f, res1_in1_val, res2_in1_val, preptup_val..., ba, x, contexts...
                )
                y_out2_val, res1_out2_val, res2_out2_val = $val_and_op!(
                    f, res1_in2_val, res2_in2_val, preptup_val..., ba, x, contexts...
                )
                res2_out1_noval = $op!(
                    f, res2_in1_noval, preptup_noval..., ba, x, contexts...
                )
                res2_out2_noval = $op!(
                    f, res2_in2_noval, preptup_noval..., ba, x, contexts...
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val === res1_out1_val
                    @test res1_in2_val === res1_out2_val
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res2_in1_val === res2_out1_val
                    @test res2_in2_val === res2_out2_val
                    @test res2_out1_val ≈ scen.res2
                    @test res2_out2_val ≈ scen.res2
                    @test res2_in1_noval === res2_out1_noval
                    @test res2_in2_noval === res2_out2_noval
                    @test res2_out1_noval ≈ scen.res2
                    @test res2_out2_noval ≈ scen.res2
                end
                if sparsity && $op == hessian
                    @test mynnz(res2_out1_val) == mynnz(scen.res2)
                    @test mynnz(res2_out2_val) == mynnz(scen.res2)
                    @test mynnz(res2_out1_noval) == mynnz(scen.res2)
                    @test mynnz(res2_out2_noval) == mynnz(scen.res2)
                end
            end
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), mysimilar(res2), prepstrict, ba, x, contexts...
            )
            @test_throws PME $op!(nothing, mysimilar(res2), prepstrict, ba, x, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, tang, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, tangrand = myrandom(x), myrandom(tang)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, tangrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.tang,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    tangrand,
                    contextsrand...,
                )
                prep_same = $prep_op_same(f, ba, x, tangrand, contexts...)
                [(), (prep,), (prepstrict,), (prep_same,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_out1_val, res1_out1_val = $val_and_op(
                    f, preptup_val..., ba, x, tang, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op(
                    f, preptup_val..., ba, x, tang, contexts...
                )
                res1_out1_noval = $op(f, preptup_noval..., ba, x, tang, contexts...)
                res1_out2_noval = $op(f, preptup_noval..., ba, x, tang, contexts...)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    for b in eachindex(scen.res1)
                        res1_out1_val[b] ≈ scen.res1[b]
                        res1_out2_val[b] ≈ scen.res1[b]
                        res1_out1_noval[b] ≈ scen.res1[b]
                        res1_out2_noval[b] ≈ scen.res1[b]
                    end
                end
            end
            @test_throws PME $val_and_op(nothing, prepstrict, ba, x, tang, contexts...)
            @test_throws PME $op(nothing, prepstrict, ba, x, tang, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, tang, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, tangrand = myrandom(x), myrandom(tang)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, tangrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.tang,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    tangrand,
                    contextsrand...,
                )
                prep_same = $prep_op_same(f, ba, x, tangrand, contexts...)
                [(), (prep,), (prepstrict,), (prep_same,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                res1_in1_val = mysimilar(res1)
                res1_in2_val = mysimilar(res1)
                res1_in1_noval = mysimilar(res1)
                res1_in2_noval = mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, res1_in1_val, preptup_val..., ba, x, tang, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, res1_in2_val, preptup_val..., ba, x, tang, contexts...
                )
                res1_out1_noval = $op!(
                    f, res1_in1_noval, preptup_noval..., ba, x, tang, contexts...
                )
                res1_out2_noval = $op!(
                    f, res1_in2_noval, preptup_noval..., ba, x, tang, contexts...
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    for b in eachindex(scen.res1)
                        @test res1_in1_val[b] === res1_out1_val[b]
                        @test res1_in2_val[b] === res1_out2_val[b]
                        @test res1_out1_val[b] ≈ scen.res1[b]
                        @test res1_out2_val[b] ≈ scen.res1[b]
                        @test res1_in1_noval[b] === res1_out1_noval[b]
                        @test res1_in2_noval[b] === res1_out2_noval[b]
                        @test res1_out1_noval[b] ≈ scen.res1[b]
                        @test res1_out2_noval[b] ≈ scen.res1[b]
                    end
                end
            end
            @test_throws PME $val_and_op!(
                nothing, mysimilar(res1), prepstrict, ba, x, tang, contexts...
            )
            @test_throws PME $op!(
                nothing, mysimilar(res1), prepstrict, ba, x, tang, contexts...
            )
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2out;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, tang, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, yrand, tangrand = myrandom(x), myrandom(y), myrandom(tang)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, copy(yrand), ba, xrand, tangrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    copy(yrand),
                    $prep_op(
                        new_smaller.f,
                        copy(new_smaller.y),
                        ba,
                        new_smaller.x,
                        new_smaller.tang,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    tangrand,
                    contextsrand...,
                )
                prep_same = $prep_op_same(f, copy(yrand), ba, x, tangrand, contexts...)
                [(), (prep,), (prepstrict,), (prep_same,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_in1_val = mysimilar(y)
                y_in2_val = mysimilar(y)
                y_in1_noval = mysimilar(y)
                y_in2_noval = mysimilar(y)
                y_out1_val, res1_out1_val = $val_and_op(
                    f, y_in1_val, preptup_val..., ba, x, tang, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op(
                    f, y_in2_val, preptup_val..., ba, x, tang, contexts...
                )
                res1_out1_noval = $op(
                    f, y_in1_noval, preptup_noval..., ba, x, tang, contexts...
                )
                res1_out2_noval = $op(
                    f, y_in2_noval, preptup_noval..., ba, x, tang, contexts...
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_in1_val === y_out1_val
                    @test y_in2_val === y_out2_val
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    for b in eachindex(scen.res1)
                        @test res1_out1_val[b] ≈ scen.res1[b]
                        @test res1_out2_val[b] ≈ scen.res1[b]
                        @test res1_out1_noval[b] ≈ scen.res1[b]
                        @test res1_out2_noval[b] ≈ scen.res1[b]
                    end
                end
            end
            @test_throws PME $val_and_op(
                nothing, mysimilar(y), prepstrict, ba, x, tang, contexts...
            )
            @test_throws PME $op(
                nothing, mysimilar(y), prepstrict, ba, x, tang, contexts...
            )
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2in;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, tang, res1, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, yrand, tangrand = myrandom(x), myrandom(y), myrandom(tang)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, copy(yrand), ba, xrand, tangrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    copy(yrand),
                    $prep_op(
                        new_smaller.f,
                        copy(new_smaller.y),
                        ba,
                        new_smaller.x,
                        new_smaller.tang,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    tangrand,
                    contextsrand...,
                )
                prep_same = $prep_op_same(f, copy(yrand), ba, x, tangrand, contexts...)
                [(), (prep,), (prepstrict,), (prep_same,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                y_in1_val, res1_in1_val = mysimilar(y), mysimilar(res1)
                y_in2_val, res1_in2_val = mysimilar(y), mysimilar(res1)
                y_in1_noval, res1_in1_noval = mysimilar(y), mysimilar(res1)
                y_in2_noval, res1_in2_noval = mysimilar(y), mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, y_in1_val, res1_in1_val, preptup_val..., ba, x, tang, contexts...
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, y_in2_val, res1_in2_val, preptup_val..., ba, x, tang, contexts...
                )
                res1_out1_noval = $op!(
                    f,
                    y_in1_noval,
                    res1_in1_noval,
                    preptup_noval...,
                    ba,
                    x,
                    tang,
                    contexts...,
                )
                res1_out2_noval = $op!(
                    f,
                    y_in2_noval,
                    res1_in2_noval,
                    preptup_noval...,
                    ba,
                    x,
                    tang,
                    contexts...,
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test y_in1_val === y_out1_val
                    @test y_in2_val === y_out2_val
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    for b in eachindex(scen.res1)
                        @test res1_in1_val[b] === res1_out1_val[b]
                        @test res1_in2_val[b] === res1_out2_val[b]
                        @test res1_out1_val[b] ≈ scen.res1[b]
                        @test res1_out2_val[b] ≈ scen.res1[b]
                        @test res1_in1_noval[b] === res1_out1_noval[b]
                        @test res1_in2_noval[b] === res1_out2_noval[b]
                        @test res1_out1_noval[b] ≈ scen.res1[b]
                        @test res1_out2_noval[b] ≈ scen.res1[b]
                    end
                end
            end
            @test_throws PME $val_and_op!(
                nothing, mysimilar(y), mysimilar(res1), prepstrict, ba, x, tang, contexts...
            )
            @test_throws PME $op!(
                nothing, mysimilar(y), mysimilar(res1), prepstrict, ba, x, tang, contexts...
            )
            scenario_intact && @test new_scen == scen
            return nothing
        end

    elseif op in [:hvp]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, tang, res1, res2, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, tangrand = myrandom(x), myrandom(tang)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, tangrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.tang,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    tangrand,
                    contextsrand...,
                )
                prep_same = $prep_op_same(f, ba, x, tangrand, contexts...)
                [(), (prep,), (prepstrict,), (prep_same,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                res2_out1_noval = $op(f, preptup_noval..., ba, x, tang, contexts...)
                res2_out2_noval = $op(f, preptup_noval..., ba, x, tang, contexts...)
                res1_out1_val, res2_out1_val = $val_and_op(
                    f, preptup_noval..., ba, x, tang, contexts...
                )
                res1_out2_val, res2_out2_val = $val_and_op(
                    f, preptup_noval..., ba, x, tang, contexts...
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    for b in eachindex(scen.res2)
                        @test res2_out1_noval[b] ≈ scen.res2[b]
                        @test res2_out2_noval[b] ≈ scen.res2[b]
                        @test res2_out1_val[b] ≈ scen.res2[b]
                        @test res2_out2_val[b] ≈ scen.res2[b]
                    end
                end
            end
            @test_throws PME $val_and_op(nothing, prepstrict, ba, x, tang, contexts...)
            @test_throws PME $op(nothing, prepstrict, ba, x, tang, contexts...)
            scenario_intact && @test new_scen == scen
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isapprox::Function,
            atol::Real,
            rtol::Real,
            scenario_intact::Bool,
            sparsity::Bool,
        )
            (; f, x, y, tang, res1, res2, contexts, smaller) = new_scen = deepcopy(scen)
            xrand, tangrand = myrandom(x), myrandom(tang)
            rewrap = Rewrap(contexts...)
            contextsrand = rewrap(map(myrandom ∘ unwrap, contexts)...)
            local prepstrict
            preptup_cands_val, preptup_cands_noval = map(1:2) do _
                new_smaller = if isnothing(smaller) || adapt_batchsize(ba, smaller) != ba
                        deepcopy(scen)
                    else
                        deepcopy(smaller)
                    end
                prep = $prep_op(f, ba, xrand, tangrand, contextsrand...)
                prepstrict = $prep_op!(
                    f,
                    $prep_op(
                        new_smaller.f,
                        ba,
                        new_smaller.x,
                        new_smaller.tang,
                        new_smaller.contexts...;
                        strict=Val(true),
                    ),
                    ba,
                    xrand,
                    tangrand,
                    contextsrand...,
                )
                prep_same = $prep_op_same(f, ba, x, tangrand, contexts...)
                [(), (prep,), (prepstrict,), (prep_same,)]
            end
            for (preptup_val, preptup_noval) in zip(preptup_cands_val, preptup_cands_noval)
                res2_in1_noval = mysimilar(res2)
                res2_in2_noval = mysimilar(res2)
                res1_in1_val, res2_in1_val = mysimilar(res1), mysimilar(res2)
                res1_in2_val, res2_in2_val = mysimilar(res1), mysimilar(res2)
                res2_out1_noval = $op!(
                    f, res2_in1_noval, preptup_noval..., ba, x, tang, contexts...
                )
                res2_out2_noval = $op!(
                    f, res2_in2_noval, preptup_noval..., ba, x, tang, contexts...
                )
                res1_out1_val, res2_out1_val = $val_and_op!(
                    f,
                    res1_in1_val,
                    res2_in1_val,
                    preptup_noval...,
                    ba,
                    x,
                    tang,
                    contexts...,
                )
                res1_out2_val, res2_out2_val = $val_and_op!(
                    f,
                    res1_in2_val,
                    res2_in2_val,
                    preptup_noval...,
                    ba,
                    x,
                    tang,
                    contexts...,
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(preptup_noval) || only(preptup_noval) isa $P
                    @test res1_in1_val === res1_out1_val
                    @test res1_in2_val === res1_out2_val
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    for b in eachindex(scen.res2)
                        @test res2_in1_noval[b] === res2_out1_noval[b]
                        @test res2_in2_noval[b] === res2_out2_noval[b]
                        @test res2_out1_noval[b] ≈ scen.res2[b]
                        @test res2_out2_noval[b] ≈ scen.res2[b]
                        @test res2_in1_val[b] === res2_out1_val[b]
                        @test res2_in2_val[b] === res2_out2_val[b]
                        @test res2_out1_val[b] ≈ scen.res2[b]
                        @test res2_out2_val[b] ≈ scen.res2[b]
                    end
                end
            end
            @test_throws PME $op!(
                nothing, mysimilar(res2), prepstrict, ba, x, tang, contexts...
            )
            @test_throws PME $val_and_op!(
                nothing,
                mysimilar(res1),
                mysimilar(res2),
                prepstrict,
                ba,
                x,
                tang,
                contexts...,
            )
            scenario_intact && @test new_scen == scen
            return nothing
        end
    end
end
