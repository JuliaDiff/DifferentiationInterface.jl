abstract type Prep{SIG} end

"""
$(docstring_preptype("PushforwardPrep", "pushforward"))
"""
abstract type PushforwardPrep{SIG} <: Prep{SIG} end

struct NoPushforwardPrep{SIG} <: PushforwardPrep{SIG}
    _sig::Val{SIG}
end

"""
$(docstring_preptype("PullbackPrep", "pullback"))
"""
abstract type PullbackPrep{SIG} <: Prep{SIG} end

struct NoPullbackPrep{SIG} <: PullbackPrep{SIG}
    _sig::Val{SIG}
end

"""
$(docstring_preptype("DerivativePrep", "derivative"))
"""
abstract type DerivativePrep{SIG} <: Prep{SIG} end

struct NoDerivativePrep{SIG} <: DerivativePrep{SIG}
    _sig::Val{SIG}
end

"""
$(docstring_preptype("GradientPrep", "gradient"))
"""
abstract type GradientPrep{SIG} <: Prep{SIG} end

struct NoGradientPrep{SIG} <: GradientPrep{SIG}
    _sig::Val{SIG}
end

"""
$(docstring_preptype("JacobianPrep", "jacobian"))
"""
abstract type JacobianPrep{SIG} <: Prep{SIG} end

struct NoJacobianPrep{SIG} <: JacobianPrep{SIG}
    _sig::Val{SIG}
end

# assume the existence of a `sparsity` field
abstract type SparseJacobianPrep{SIG} <: JacobianPrep{SIG} end

"""
$(docstring_preptype("HVPPrep", "hvp"))
"""
abstract type HVPPrep{SIG} <: Prep{SIG} end

struct NoHVPPrep{SIG} <: HVPPrep{SIG}
    _sig::Val{SIG}
end

"""
$(docstring_preptype("HessianPrep", "hessian"))
"""
abstract type HessianPrep{SIG} <: Prep{SIG} end

struct NoHessianPrep{SIG} <: HessianPrep{SIG}
    _sig::Val{SIG}
end

# assume the existence of a `sparsity` field
abstract type SparseHessianPrep{SIG} <: HessianPrep{SIG} end

"""
$(docstring_preptype("SecondDerivativePrep", "second_derivative"))
"""
abstract type SecondDerivativePrep{SIG} <: Prep{SIG} end

struct NoSecondDerivativePrep{SIG} <: SecondDerivativePrep{SIG}
    _sig::Val{SIG}
end

## Checks

is_strict(::Prep{Nothing}) = Val(false)
is_strict(::Prep) = Val(true)

struct PreparationMismatchError{SIG,EXEC_SIG} <: Exception
    format::Vector{Symbol}
end

function PreparationMismatchError(
    ::Type{SIG}, ::Type{EXEC_SIG}; format
) where {SIG,EXEC_SIG}
    return PreparationMismatchError{SIG,EXEC_SIG}(format)
end

function Base.showerror(
    io::IO, e::PreparationMismatchError{SIG,EXEC_SIG}
) where {SIG<:Tuple,EXEC_SIG<:Tuple}
    println(
        io,
        "PreparationMismatchError (inconsistent types between preparation and execution):",
    )
    for (s, pt, et) in zip(e.format, SIG.types, EXEC_SIG.types)
        if pt == et
            println(io, "  - $s: ✅")
        else
            println(io, "  - $s: ❌\n    - prep: $pt\n    - exec: $et")
        end
    end
    println(
        io,
        "If you are confident that this check is superfluous, you can disable it by running preparation with the keyword argument `strict=Val(false)` inside DifferentiationInterface.",
    )
    return nothing
end

function signature(
    f, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Val{S}
) where {C,S}
    if S
        return Val(typeof((f, backend, x, contexts)))
    else
        return Val(Nothing)
    end
end

function signature(
    f!, y, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Val{S}
) where {C,S}
    if S
        return Val(typeof((f!, y, backend, x, contexts)))
    else
        return Val(Nothing)
    end
end

function signature(
    f, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}; strict::Val{S}
) where {C,S}
    if S
        return Val(typeof((f, backend, x, t, contexts)))
    else
        return Val(Nothing)
    end
end

function signature(
    f!,
    y,
    backend::AbstractADType,
    x,
    t::NTuple,
    contexts::Vararg{Context,C};
    strict::Val{S},
) where {C,S}
    if S
        return Val(typeof((f!, y, backend, x, t, contexts)))
    else
        return Val(Nothing)
    end
end

function check_prep(
    f, ::Prep{SIG}, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        EXEC_SIG = typeof((f, backend, x, contexts))
        if SIG != EXEC_SIG
            throw(
                PreparationMismatchError(
                    SIG, EXEC_SIG; format=[:f, :backend, :x, :contexts]
                ),
            )
        end
    end
end

function check_prep(
    f!, y, ::Prep{SIG}, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        EXEC_SIG = typeof((f!, y, backend, x, contexts))
        if SIG != EXEC_SIG
            throw(
                PreparationMismatchError(
                    SIG, EXEC_SIG; format=[:f!, :y, :backend, :x, :contexts]
                ),
            )
        end
    end
end

function check_prep(
    f, ::Prep{SIG}, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        EXEC_SIG = typeof((f, backend, x, t, contexts))
        if SIG != EXEC_SIG
            throw(
                PreparationMismatchError(
                    SIG, EXEC_SIG; format=[:f, :backend, :x, :tang, :contexts]
                ),
            )
        end
    end
end

function check_prep(
    f!, y, ::Prep{SIG}, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        EXEC_SIG = typeof((f!, y, backend, x, t, contexts))
        if SIG != EXEC_SIG
            throw(
                PreparationMismatchError(
                    SIG, EXEC_SIG; format=[:f!, :y, :backend, :x, :tang, :contexts]
                ),
            )
        end
    end
end
