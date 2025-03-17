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
        :gradient_and_hvp
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

    if op in (:derivative, :jacobian, :gradient)
        # 1-arg
        @eval function $op(
            f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $op(f, prep, backend, x, contexts...)
        end
        @eval function $op!(
            f::F, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $op!(f, result, prep, backend, x, contexts...)
        end
        @eval function $val_and_op(
            f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $val_and_op(f, prep, backend, x, contexts...)
        end
        @eval function $val_and_op!(
            f::F, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $val_and_op!(f, result, prep, backend, x, contexts...)
        end
        op == :gradient && continue
        # 2-arg
        @eval function $op(
            f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, contexts...)
            return $op(f!, y, prep, backend, x, contexts...)
        end
        @eval function $op!(
            f!::F, y, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, contexts...)
            return $op!(f!, y, result, prep, backend, x, contexts...)
        end
        @eval function $val_and_op(
            f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, contexts...)
            return $val_and_op(f!, y, prep, backend, x, contexts...)
        end
        @eval function $val_and_op!(
            f!::F, y, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, contexts...)
            return $val_and_op!(f!, y, result, prep, backend, x, contexts...)
        end

    elseif op in (:second_derivative, :hessian)
        # 1-arg
        @eval function $op(
            f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $op(f, prep, backend, x, contexts...)
        end
        @eval function $op!(
            f::F, result2, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $op!(f, result2, prep, backend, x, contexts...)
        end
        @eval function $val_and_op(
            f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $val_and_op(f, prep, backend, x, contexts...)
        end
        @eval function $val_and_op!(
            f::F, result1, result2, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, contexts...)
            return $val_and_op!(f, result1, result2, prep, backend, x, contexts...)
        end

    elseif op in (:pushforward, :pullback, :hvp)
        @eval function $op(
            f::F, backend::AbstractADType, x, tang::NTuple, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, tang, contexts...)
            return $op(f, prep, backend, x, tang, contexts...)
        end
        @eval function $op!(
            f::F,
            result::NTuple,
            backend::AbstractADType,
            x,
            tang::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, tang, contexts...)
            return $op!(f, result, prep, backend, x, tang, contexts...)
        end
        @eval function $val_and_op(
            f::F, backend::AbstractADType, x, tang::NTuple, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f, backend, x, tang, contexts...)
            return $val_and_op(f, prep, backend, x, tang, contexts...)
        end

        if op in (:pushforward, :pullback)
            @eval function $val_and_op!(
                f::F,
                result::NTuple,
                backend::AbstractADType,
                x,
                tang::NTuple,
                contexts::Vararg{Context,C},
            ) where {F,C}
                prep = $prep_op(Val(true), f, backend, x, tang, contexts...)
                return $val_and_op!(f, result, prep, backend, x, tang, contexts...)
            end
        elseif op == :hvp
            @eval function $val_and_op!(
                f::F,
                result1,
                result2::NTuple,
                backend::AbstractADType,
                x,
                tang::NTuple,
                contexts::Vararg{Context,C},
            ) where {F,C}
                prep = $prep_op(Val(true), f, backend, x, tang, contexts...)
                return $val_and_op!(
                    f, result1, result2, prep, backend, x, tang, contexts...
                )
            end
        end

        op == :hvp && continue

        @eval function $op(
            f!::F, y, backend::AbstractADType, x, tang::NTuple, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, tang, contexts...)
            return $op(f!, y, prep, backend, x, tang, contexts...)
        end
        @eval function $op!(
            f!::F,
            y,
            result::NTuple,
            backend::AbstractADType,
            x,
            tang::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, tang, contexts...)
            return $op!(f!, y, result, prep, backend, x, tang, contexts...)
        end
        @eval function $val_and_op(
            f!::F, y, backend::AbstractADType, x, tang::NTuple, contexts::Vararg{Context,C}
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, tang, contexts...)
            return $val_and_op(f!, y, prep, backend, x, tang, contexts...)
        end
        @eval function $val_and_op!(
            f!::F,
            y,
            result::NTuple,
            backend::AbstractADType,
            x,
            tang::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            prep = $prep_op(Val(true), f!, y, backend, x, tang, contexts...)
            return $val_and_op!(f!, y, result, prep, backend, x, tang, contexts...)
        end
    end
end
