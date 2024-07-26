module DifferentiationInterfaceEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    NoDerivativeExtras,
    NoGradientExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    NoPushforwardExtras,
    pick_batchsize
using DocStringExtensions
using Enzyme:
    Active,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    Forward,
    ForwardMode,
    MixedDuplicated,
    Mode,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitWithPrimal,
    ReverseMode,
    autodiff,
    autodiff_deferred,
    autodiff_deferred_thunk,
    autodiff_thunk,
    chunkedonehot,
    gradient,
    gradient!,
    guess_activity,
    jacobian,
    make_zero,
    make_zero!,
    onehot

struct AutoDeferredEnzyme{M,constant_function} <: ADTypes.AbstractADType
    mode::M
end

ADTypes.mode(backend::AutoDeferredEnzyme) = ADTypes.mode(AutoEnzyme(backend.mode))

function DI.nested(backend::AutoEnzyme{M,constant_function}) where {M,constant_function}
    return AutoDeferredEnzyme{M,constant_function}(backend.mode)
end

const AnyAutoEnzyme{M,constant_function} = Union{
    AutoEnzyme{M,constant_function},AutoDeferredEnzyme{M,constant_function}
}

# forward mode if possible
forward_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
forward_mode(::AnyAutoEnzyme{Nothing}) = Forward

# reverse mode if possible
reverse_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
reverse_mode(::AnyAutoEnzyme{Nothing}) = Reverse

DI.check_available(::AutoEnzyme) = true

# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
DI.pick_batchsize(::AnyAutoEnzyme, dimension::Integer) = min(dimension, 16)

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basis(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,true}) where {M}
    return Const(f)
end

function get_f_and_df(f, backend::AnyAutoEnzyme{M,false}) where {M}
    mode = isnothing(backend.mode) ? Reverse : backend.mode
    A = guess_activity(typeof(f), mode)
    if A <: Const || A <: Active
        return Const(f)
    elseif A <: Duplicated || A <: DuplicatedNoNeed || A <: MixedDuplicated
        df = make_zero(f)
        return Duplicated(f, df)
    else
        error("Unexpected activity guessed for the function `f`.")
    end
end

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

end # module
