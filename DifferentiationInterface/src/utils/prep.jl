abstract type Prep{SIG} end

"""
$(docstring_preptype("PushforwardPrep", "pushforward"))
"""
abstract type PushforwardPrep{SIG} <: Prep{SIG} end
struct NoPushforwardPrep{SIG} <: PushforwardPrep{SIG} end

"""
$(docstring_preptype("PullbackPrep", "pullback"))
"""
abstract type PullbackPrep{SIG} <: Prep{SIG} end
struct NoPullbackPrep{SIG} <: PullbackPrep{SIG} end

"""
$(docstring_preptype("DerivativePrep", "derivative"))
"""
abstract type DerivativePrep{SIG} <: Prep{SIG} end
struct NoDerivativePrep{SIG} <: DerivativePrep{SIG} end

"""
$(docstring_preptype("GradientPrep", "gradient"))
"""
abstract type GradientPrep{SIG} <: Prep{SIG} end
struct NoGradientPrep{SIG} <: GradientPrep{SIG} end

"""
$(docstring_preptype("JacobianPrep", "jacobian"))
"""
abstract type JacobianPrep{SIG} <: Prep{SIG} end
struct NoJacobianPrep{SIG} <: JacobianPrep{SIG} end

"""
$(docstring_preptype("HVPPrep", "hvp"))
"""
abstract type HVPPrep{SIG} <: Prep{SIG} end
struct NoHVPPrep{SIG} <: HVPPrep{SIG} end

"""
$(docstring_preptype("HessianPrep", "hessian"))
"""
abstract type HessianPrep{SIG} <: Prep{SIG} end
struct NoHessianPrep{SIG} <: HessianPrep{SIG} end

"""
$(docstring_preptype("SecondDerivativePrep", "second_derivative"))
"""
abstract type SecondDerivativePrep{SIG} <: Prep{SIG} end
struct NoSecondDerivativePrep{SIG} <: SecondDerivativePrep{SIG} end

## Checks

is_strict(::Prep{SIG}) where {SIG} = SIG !== Nothing

function signature(
    f, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Bool
) where {C}
    if strict
        return typeof((f, backend, x, contexts))
    else
        return Nothing
    end
end

function signature(
    f!, y, backend::AbstractADType, x, contexts::Vararg{Context,C}; strict::Bool
) where {C}
    if strict
        return typeof((f!, y, backend, x, contexts))
    else
        return Nothing
    end
end

function signature(
    f, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}; strict::Bool
) where {C}
    if strict
        return typeof((f, backend, x, t, contexts))
    else
        return Nothing
    end
end

function signature(
    f!, y, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}; strict::Bool
) where {C}
    if strict
        return typeof((f!, y, backend, x, t, contexts))
    else
        return Nothing
    end
end

function check_prep(
    f, ::Prep{SIG}, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        @assert SIG == typeof((f, backend, x, contexts))
    end
end

function check_prep(
    f!, y, ::Prep{SIG}, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        @assert SIG == typeof((f!, y, backend, x, contexts))
    end
end

function check_prep(
    f, ::Prep{SIG}, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        @assert SIG == typeof((f, backend, x, t, contexts))
    end
end

function check_prep(
    f!, y, ::Prep{SIG}, backend::AbstractADType, x, t::NTuple, contexts::Vararg{Context,C}
) where {SIG,C}
    if SIG !== Nothing
        @assert SIG == typeof((f!, y, backend, x, t, contexts))
    end
end
