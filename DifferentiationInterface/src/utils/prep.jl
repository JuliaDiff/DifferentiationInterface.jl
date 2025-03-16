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

struct PreparationMismatchError{SIG,RUNTIME_SIG} <: Exception end

function PreparationMismatchError(::Type{SIG}, ::Type{RUNTIME_SIG}) where {SIG,RUNTIME_SIG}
    return PreparationMismatchError{SIG,RUNTIME_SIG}()
end

function Base.showerror(
    io::IO, e::PreparationMismatchError{SIG,RUNTIME_SIG}
) where {SIG,RUNTIME_SIG}
    msg = """
    Inconsistent signatures:
     - at preparation time: $SIG
     - at execution time: $RUNTIME_SIG
    """
    return print(io, msg)
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
        RUNTIME_SIG = typeof((f, backend, x, contexts))
        if SIG != RUNTIME_SIG
            throw(PreparationMismatchError(SIG, RUNTIME_SIG))
        end
    end
end

function check_prep(
    f!, y, ::Prep{SIG}, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        RUNTIME_SIG = typeof((f!, y, backend, x, contexts))
        if SIG != RUNTIME_SIG
            throw(PreparationMismatchError(SIG, RUNTIME_SIG))
        end
    end
end

function check_prep(
    f, ::Prep{SIG}, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        RUNTIME_SIG = typeof((f, backend, x, t, contexts))
        if SIG != RUNTIME_SIG
            throw(PreparationMismatchError(SIG, RUNTIME_SIG))
        end
    end
end

function check_prep(
    f!, y, ::Prep{SIG}, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        RUNTIME_SIG = typeof((f!, y, backend, x, t, contexts))
        if SIG != RUNTIME_SIG
            throw(PreparationMismatchError(SIG, RUNTIME_SIG))
        end
    end
end
