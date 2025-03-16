for op in [
    :derivative,
    :gradient,
    :jacobian,
    :second_derivative,
    :hessian,
    :pushforward,
    :pullback,
    :hvp,
]
    op! = Symbol(op, "!")
    val_and_op = if op == :second_derivative
        :value_derivative_and_second_derivative
    elseif op == :hessian
        :value_gradient_and_hessian
    elseif op == :hvp
        nothing
    else
        Symbol("value_and_", op)
    end
    val_and_op! = Symbol(val_and_op, "!")
    prep_op = Symbol("prepare_", op)
    prep_op! = Symbol("prepare!_", op)
    prep_op_same_point = Symbol("prepare_", op, "_same_point")
    P = if op == :derivative
        DerivativePrep
    elseif op == :gradient
        GradientPrep
    elseif op == :jacobian
        JacobianPrep
    elseif op == :second_derivative
        SecondDerivativePrep
    elseif op == :hessian
        HessianPrep
    elseif op == :pushforward
        PushforwardPrep
    elseif op == :pullback
        PullbackPrep
    elseif op == :hvp
        HVPPrep
    end

    if op in (:derivative, :gradient, :jacobian)
        # 1-arg
        @eval function $prep_op!(
            f::F, old_prep::$P, backend::AbstractADType, x, contexts::Vararg{Context,C};
        ) where {F,C}
            check_prep(f, old_prep, backend, x, contexts...)
            return $prep_op(f, backend, x, contexts...; strict=is_strict(old_prep))
        end
        op == :gradient && continue
        # 2-arg
        @eval function $prep_op!(
            f!::F, y, old_prep::$P, backend::AbstractADType, x, contexts::Vararg{Context,C};
        ) where {F,C}
            check_prep(f!, y, old_prep, backend, x, contexts...)
            return $prep_op(f!, y, backend, x, contexts...; strict=is_strict(old_prep))
        end

    elseif op in (:second_derivative, :hessian)
        # 1-arg
        @eval function $prep_op!(
            f::F, old_prep::$P, backend::AbstractADType, x, contexts::Vararg{Context,C};
        ) where {F,C}
            check_prep(f, old_prep, backend, x, contexts...)
            return $prep_op(f, backend, x, contexts...; strict=is_strict(old_prep))
        end

    elseif op in (:pushforward, :pullback, :hvp)
        # 1-arg
        @eval function $prep_op!(
            f::F,
            old_prep::$P,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C};
        ) where {F,C}
            check_prep(f, old_prep, backend, x, seed, contexts...)
            return $prep_op(f, backend, x, seed, contexts...; strict=is_strict(old_prep))
        end
        @eval function $prep_op_same_point(
            f::F,
            prep::$P,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            check_prep(f, prep, backend, x, seed, contexts...)
            return prep
        end
        @eval function $prep_op_same_point(
            f::F,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C};
            strict::Val=Val(false),
        ) where {F,C}
            prep = $prep_op(f, backend, x, seed, contexts...; strict)
            return $prep_op_same_point(f, prep, backend, x, seed, contexts...)
        end
        op == :hvp && continue
        # 2-arg
        @eval function $prep_op!(
            f!::F,
            y,
            old_prep::$P,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            check_prep(f!, y, old_prep, backend, x, seed, contexts...)
            return $prep_op(
                f!, y, backend, x, seed, contexts...; strict=is_strict(old_prep)
            )
        end
        @eval function $prep_op_same_point(
            f!::F,
            y,
            prep::$P,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            check_prep(f!, y, prep, backend, x, seed, contexts...)
            return prep
        end
        @eval function $prep_op_same_point(
            f!::F,
            y,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C};
            strict::Val=Val(false),
        ) where {F,C}
            prep = $prep_op(f!, y, backend, x, seed, contexts...; strict)
            return $prep_op_same_point(f!, y, prep, backend, x, seed, contexts...)
        end
    end
end
