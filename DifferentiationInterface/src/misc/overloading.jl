"""
    overloaded_input_type(prep)

If it exists, return the overloaded input type which will be passed to the differentiated function when preparation result `prep` is reused.

!!! danger
    This function is experimental and not part of the public API.
"""
function overloaded_input_type end

function overloaded_input(::typeof(pushforward), f, backend::AbstractADType, x, tx::NTuple)
    throw(ArgumentError("Overloaded input not defined"))
end

function overloaded_input(
    ::typeof(pushforward), f!, y, backend::AbstractADType, x, tx::NTuple
)
    throw(ArgumentError("Overloaded input not defined"))
end
