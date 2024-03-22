abstract type AbstractOperator end
abstract type AbstractFirstOrderOperator <: AbstractOperator end
abstract type AbstractSecondOrderOperator <: AbstractOperator end

# First-Order Operators
struct Pullback{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Pushforward{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Gradient{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Multiderivative{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Jacobian{M<:MutationBehavior} <: AbstractFirstOrderOperator end
struct Derivative{M<:MutationBehavior} <: AbstractFirstOrderOperator end

const PullbackAllocating        = Pullback{MutationNotSupported}
const PullbackMutating          = Pullback{MutationSupported}
const PushforwardAllocating     = Pushforward{MutationNotSupported}
const PushforwardMutating       = Pushforward{MutationSupported}
const GradientAllocating        = Gradient{MutationNotSupported}
const GradientMutating          = Gradient{MutationSupported}
const MultiderivativeAllocating = Multiderivative{MutationNotSupported}
const MultiderivativeMutating   = Multiderivative{MutationSupported}
const JacobianAllocating        = Jacobian{MutationNotSupported}
const JacobianMutating          = Jacobian{MutationSupported}
const DerivativeAllocating      = Derivative{MutationNotSupported}
const DerivativeMutating        = Derivative{MutationSupported}

# Second-order operators
struct SecondDerivative{M<:MutationBehavior} <: AbstractSecondOrderOperator end
struct Hessian{M<:MutationBehavior} <: AbstractSecondOrderOperator end
struct HessianVectorProduct{M<:MutationBehavior} <: AbstractSecondOrderOperator end

const SecondDerivativeAllocating     = SecondDerivative{MutationNotSupported}
const SecondDerivativeMutating       = SecondDerivative{MutationSupported}
const HessianAllocating              = Hessian{MutationNotSupported}
const HessianMutating                = Hessian{MutationSupported}
const HessianVectorProductAllocating = HessianVectorProduct{MutationNotSupported}
const HessianVectorProductMutating   = HessianVectorProduct{MutationSupported}

## Utilities
# order
isfirstorder(::AbstractOperator)           = false
isfirstorder(::AbstractFirstOrderOperator) = true

issecondorder(::AbstractOperator)            = false
issecondorder(::AbstractSecondOrderOperator) = true

# allocations
ismutating(::Type{<:MutationBehavior}) = false
ismutating(::Type{MutationSupported})  = true

ismutating(::Pullback{M}) where {M}             = ismutating(M)
ismutating(::Pushforward{M}) where {M}          = ismutating(M)
ismutating(::Gradient{M}) where {M}             = ismutating(M)
ismutating(::Multiderivative{M}) where {M}      = ismutating(M)
ismutating(::Derivative{M}) where {M}           = ismutating(M)
ismutating(::Jacobian{M}) where {M}             = ismutating(M)
ismutating(::SecondDerivative{M}) where {M}     = ismutating(M)
ismutating(::Hessian{M}) where {M}              = ismutating(M)
ismutating(::HessianVectorProduct{M}) where {M} = ismutating(M)

isallocating(op) = !ismutating(op)

# input-output compatibility
iscompatible(op::AbstractOperator, x, y) = false
iscompatible(op::Pullback, x, y)         = true
iscompatible(op::Pushforward, x, y)      = true

iscompatible(op::Gradient, x::AbstractArray, y::Number)             = true
iscompatible(op::Multiderivative, x::Number, y::AbstractArray)      = true
iscompatible(op::Derivative, x::Number, y::Number)                  = true
iscompatible(op::Jacobian, x::AbstractArray, y::AbstractArray)      = true
iscompatible(op::SecondDerivative, x::Number, y::Number)            = true
iscompatible(op::Hessian, x::AbstractArray, y::Number)              = true
iscompatible(op::HessianVectorProduct, x::AbstractArray, y::Number) = true

## Pretty-printing
alloc_string(T::Type{<:MutationBehavior}) = ismutating(T) ? "mutating" : "allocating"

Base.string(::Pullback{M}) where {M}             = "Pullback, $(alloc_string(M))"
Base.string(::Pushforward{M}) where {M}          = "Pushforward, $(alloc_string(M))"
Base.string(::Gradient{M}) where {M}             = "Gradient, $(alloc_string(M))"
Base.string(::Multiderivative{M}) where {M}      = "Multiderivative, $(alloc_string(M))"
Base.string(::Derivative{M}) where {M}           = "Derivative, $(alloc_string(M))"
Base.string(::Jacobian{M}) where {M}             = "Jacobian, $(alloc_string(M))"
Base.string(::SecondDerivative{M}) where {M}     = "Second derivative, $(alloc_string(M))"
Base.string(::Hessian{M}) where {M}              = "Hessian, $(alloc_string(M))"
Base.string(::HessianVectorProduct{M}) where {M} = "Hessian-vector product, $(alloc_string(M))"

## Convert symbols to traits
const OPERATOR_SYMBOL_TO_TRAIT = Dict(
    :pushforward_allocating            => PushforwardAllocating(),
    :pushforward_mutating              => PushforwardMutating(),
    :pullback_allocating               => PullbackAllocating(),
    :pullback_mutating                 => PullbackMutating(),
    :multiderivative_allocating        => MultiderivativeAllocating(),
    :multiderivating_mutating          => MultiderivativeMutating(),
    :derivative_allocating             => DerivativeAllocating(),
    :derivative_mutating               => DerivativeMutating(),
    :gradient_allocating               => GradientAllocating(),
    :gradient_mutating                 => GradientMutating(),
    :jacobian_allocating               => JacobianAllocating(),
    :jacobian_mutating                 => JacobianMutating(),
    :second_derivative_allocating      => SecondDerivativeAllocating(),
    :second_derivative_mutating        => SecondDerivativeMutating(),
    :hessian_allocating                => HessianAllocating(),
    :hessian_mutating                  => HessianMutating(),
    :hessian_vector_product_allocating => HessianVectorProductAllocating(),
    :hessian_vector_product_mutating   => HessianVectorProductMutating(),
)

operator_trait(op::AbstractOperator) = op
function operator_trait(sym::Symbol)
    !haskey(OPERATOR_SYMBOL_TO_TRAIT, sym) &&
        throw(ArgumentError("Invalid operator symbol: $sym"))
    return OPERATOR_SYMBOL_TO_TRAIT[sym]
end
