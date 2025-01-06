module DifferentiationInterfaceTestJLArraysExt

import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using JLArrays: JLArray, JLVector, JLMatrix, jl

myjl(f::Function) = f
function myjl(::DIT.NumToArr{A}) where {T,N,A<:AbstractArray{T,N}}
    return DIT.NumToArr(JLArray{T,N})
end

function (f::DIT.NumToArr{JLVector{T}})(x::Number) where {T}
    a = JLVector{T}(Vector(1:6))  # avoid mutation
    return sin.(x .* a)
end

function (f::DIT.NumToArr{JLMatrix{T}})(x::Number) where {T}
    a = JLMatrix{T}(Matrix(reshape(1:6, 2, 3)))  # avoid mutation
    return sin.(x .* a)
end

myjl(f::DIT.FunctionModifier) = f

myjl(x::Number) = x
myjl(x::AbstractArray) = jl(x)
myjl(x::Tuple) = map(myjl, x)
myjl(x::DI.Constant) = DI.Constant(myjl(DI.unwrap(x)))
myjl(x::DI.Cache) = DI.Cache(myjl(DI.unwrap(x)))
myjl(::Nothing) = nothing

function myjl(scen::DIT.Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, tang, contexts, res1, res2) = scen
    return DIT.Scenario{op,pl_op,pl_fun}(
        myjl(f);
        x=myjl(x),
        y=myjl(y),
        tang=myjl(tang),
        contexts=myjl(contexts),
        res1=myjl(res1),
        res2=myjl(res2),
    )
end

function DIT.gpu_scenarios(args...; kwargs...)
    scens = DIT.default_scenarios(args...; kwargs...)
    return myjl.(scens)
end

end
