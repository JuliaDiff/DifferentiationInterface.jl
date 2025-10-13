function docstring_preptype(preptype::AbstractString, operator::AbstractString)
    return """
        $(preptype)

    Abstract type for additional information needed by [`$(operator)`](@ref) and its variants.
    """
end

function samepoint_warning(samepoint::Bool)
    return if samepoint
        ", _if they are applied at the same point `x` and with the same `contexts`_"
    else
        ""
    end
end

function docstring_prepare(operator; samepoint = false, inplace = false)
    return """
    Create a `prep` object that can be given to [`$(operator)`](@ref) and its variants to speed them up$(samepoint_warning(samepoint)).

    Depending on the backend, this can have several effects (preallocating memory, recording an execution trace) which are transparent to the user.
    $(inplace ? "\nFor in-place functions, `y` is mutated by `f!` during preparation." : "")

    !!! warning
        The preparation result `prep` is only reusable as long as the arguments to `$operator` do not change type or size, and the function and backend themselves are not modified.
        Otherwise, preparation becomes invalid and you need to run it again.
        In some settings, invalid preparations may still give correct results (e.g. for backends that require no preparation), but this is not a semantic guarantee and should not be relied upon.

    !!! danger
        The preparation result `prep` is **not thread-safe**. Sharing it between threads may lead to unexpected behavior. If you need to run differentiation concurrently, prepare separate `prep` objects for each thread.

    When `strict=Val(true)` (the default), type checking is enforced between preparation and execution (but size checking is left to the user).
    While your code may work for different types by setting `strict=Val(false)`, this is not guaranteed by the API and can break without warning.
    """
end

function docstring_prepare!(operator)
    return """
    Same behavior as [`prepare_$(operator)`](@ref) but can resize the contents of an existing `prep` object to avoid some allocations.

    There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

    !!! danger
        Compared to when `prep` was first created, the only authorized modification is a size change for input `x` or output `y`.
        Any other modification (like a change of type for the input) is not supported and will give erroneous results.

    !!! danger
        For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
    """
end

function docstring_preparation_hint(operator::AbstractString; same_point = false)
    if same_point
        return "To improve performance via operator preparation, refer to [`prepare_$(operator)`](@ref) and [`prepare_$(operator)_same_point`](@ref)."
    else
        return "To improve performance via operator preparation, refer to [`prepare_$(operator)`](@ref)."
    end
end
