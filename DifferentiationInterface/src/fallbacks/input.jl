function overloaded_input end
function overloaded_input_type end

function error_if_overloading(backend)
    if check_operator_overloading(backend)
        throw(
            ArgumentError(
                "The current backend is based on operator overloading, a custom method for `overloaded_input` is therefore necessary. Please open an issue on DifferentiationInterface.jl if you encounter this error.",
            ),
        )
    end
end

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
    if op in (:derivative, :jacobian, :gradient)
        @eval function overloaded_input(
            ::typeof($op), f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            error_if_overloading(backend)
            return copy(x)
        end
        op == :gradient && continue
        @eval function overloaded_input(
            ::typeof($op), f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            error_if_overloading(backend)
            return copy(x)
        end

    elseif op in (:second_derivative, :hessian)
        @eval function overloaded_input(
            ::typeof($op), f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
        ) where {F,C}
            error_if_overloading(backend)
            return copy(x)
        end

    elseif op in (:pushforward, :pullback, :hvp)
        @eval function overloaded_input(
            ::typeof($op),
            f::F,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            error_if_overloading(backend)
            return copy(x)
        end
        op == :hvp && continue
        @eval function overloaded_input(
            ::typeof($op),
            f!::F,
            y,
            backend::AbstractADType,
            x,
            seed::NTuple,
            contexts::Vararg{Context,C},
        ) where {F,C}
            error_if_overloading(backend)
            return copy(x)
        end
    end
end
