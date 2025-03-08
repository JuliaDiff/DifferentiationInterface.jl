abstract type Prep{F,Y,B<:AbstractADType,X,T<:Union{NTuple,Nothing},CC} end

abstract type PushforwardPrep{F,Y,B,X,T,CC} <: Prep{F,Y,B,X,T,CC} end
struct NoPushforwardPrep{F,Y,B,X,T,CC} <: PushforwardPrep{F,Y,B,X,T,CC} end

abstract type PullbackPrep{F,Y,B,X,T,CC} <: Prep{F,Y,B,X,T,CC} end
struct NoPullbackPrep{F,Y,B,X,T,CC} <: PullbackPrep{F,Y,B,X,T,CC} end

abstract type DerivativePrep{F,Y,B,X,CC} <: Prep{F,Y,B,X,Nothing,CC} end
struct NoDerivativePrep{F,Y,B,X,CC} <: DerivativePrep{F,Y,B,X,CC} end

abstract type GradientPrep{F,B,X,CC} <: Prep{F,Nothing,B,X,Nothing,CC} end
struct NoGradientPrep{F,B,X,CC} <: GradientPrep{F,B,X,CC} end

abstract type JacobianPrep{F,Y,B,X,CC} <: Prep{F,Y,B,X,Nothing,CC} end
struct NoJacobianPrep{F,Y,B,X,CC} <: JacobianPrep{F,Y,B,X,CC} end

abstract type HVPPrep{F,B,X,T,CC} <: Prep{F,Nothing,B,X,T,CC} end
struct NoHVPPrep{F,B,X,T,CC} <: HVPPrep{F,B,X,T,CC} end

abstract type HessianPrep{F,B,X,CC} <: Prep{F,Nothing,B,X,Nothing,CC} end
struct NoHessianPrep{F,B,X,CC} <: HessianPrep{F,B,X,CC} end

abstract type SecondDerivativePrep{F,Y,B,X,CC} <: Prep{F,Y,B,X,Nothing,CC} end
struct NoSecondDerivativePrep{F,Y,B,X,CC} <: SecondDerivativePrep{F,Y,B,X,CC} end

function check_prep(
    f, ::Prep{F,Y,B,X,T,CC}, backend, x, contexts::Vararg{Context,C}
) where {F,Y,B,X,T,CC,C}
    @assert typeof(f) == F
    @assert Nothing == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert Nothing == T
    @assert typeof(contexts) == CC
end

function check_prep(
    f, ::Prep{F,Y,B,X,T,CC}, backend, x, t::NTuple, contexts::Vararg{Context,C}
) where {F,Y,B,X,T,CC,C}
    @assert typeof(f) == F
    @assert Nothing == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert typeof(t) == T
    @assert typeof(contexts) == CC
end

function check_prep(
    f, y, ::Prep{F,Y,B,X,T,CC}, backend, x, contexts::Vararg{Context,C}
) where {F,Y,B,X,T,CC,C}
    @assert typeof(f) == F
    @assert typeof(y) == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert Nothing == T
    @assert typeof(contexts) == CC
end

function check_prep(
    f, y, ::Prep{F,Y,B,X,CC}, backend, x, t::NTuple, contexts::Vararg{Context,C}
) where {F,Y,B,X,CC,C}
    @assert typeof(f) == F
    @assert typeof(y) == Y
    @assert typeof(backend) == B
    @assert typeof(x) == X
    @assert typeof(t) == T
    @assert typeof(contexts) == CC
end
