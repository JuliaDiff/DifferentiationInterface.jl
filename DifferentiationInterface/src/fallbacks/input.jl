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

for op in [:pushforward, :pullback]
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
end
