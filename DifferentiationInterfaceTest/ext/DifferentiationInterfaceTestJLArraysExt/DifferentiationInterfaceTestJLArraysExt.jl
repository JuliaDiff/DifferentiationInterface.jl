module DifferentiationInterfaceTestJLArraysExt

import DifferentiationInterface as DI
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using JLArrays: JLArray, jl
using Random: AbstractRNG, default_rng

myjl(f::Function) = f
function myjl(::DIT.NumToArr{A}) where {T,N,A<:AbstractArray{T,N}}
    return DIT.NumToArr(JLArray{T,N})
end

myjl(f::DIT.MultiplyByConstant) = f
myjl(f::DIT.WritableClosure) = f

myjl(x::Number) = x
myjl(x::AbstractArray) = jl(x)
myjl(x::Tuple) = map(myjl, x)
myjl(x::DI.Constant) = DI.Constant(myjl(DI.unwrap(x)))
myjl(::Nothing) = nothing

function myjl(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, tang, contexts, res1, res2) = scen
    return Scenario{op,pl_op,pl_fun}(
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
