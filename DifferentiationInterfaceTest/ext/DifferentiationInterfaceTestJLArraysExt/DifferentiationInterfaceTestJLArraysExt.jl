module DifferentiationInterfaceTestJLArraysExt

import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using JLArrays: JLArray, JLVector, JLMatrix, jl

jl_num_to_vec(x::Number) = sin.(jl([1, 2]) .* x)
jl_num_to_mat(x::Number) = hcat(jl_num_to_vec(x), jl_num_to_vec(3x))

const NTV = typeof(DIT.num_to_vec)
const NTM = typeof(DIT.num_to_mat)
myjl(f::Function) = f
myjl(::NTV) = jl_num_to_vec
myjl(::NTM) = jl_num_to_mat
myjl(f::DIT.FunctionModifier) = f

myjl(x::Number) = x
myjl(x::AbstractArray) = jl(x)
myjl(x::Tuple) = map(myjl, x)
myjl(x::DI.Constant) = DI.Constant(myjl(DI.unwrap(x)))
myjl(x::DI.Cache{<:AbstractArray}) = DI.Cache(myjl(DI.unwrap(x)))
myjl(x::DI.Cache{<:Union{Tuple,NamedTuple}}) = map(myjl, map(DI.Cache, DI.unwrap(x)))
myjl(::Nothing) = nothing

function myjl(scen::DIT.Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, t, contexts, prep_args, res1, res2) = scen
    return DIT.Scenario{op,pl_op,pl_fun}(;
        f=myjl(f),
        x=myjl(x),
        y=myjl(y),
        t=myjl(t),
        contexts=myjl(contexts),
        prep_args=map(myjl, prep_args),
        res1=myjl(res1),
        res2=myjl(res2),
    )
end

function DIT.gpu_scenarios(args...; kwargs...)
    scens = DIT.default_scenarios(args...; kwargs...)
    return myjl.(scens)
end

end
