module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
using DifferentiationInterface: myupdate!!
import DifferentiationInterface as DI
using FillArrays: OneElement
using FiniteDifferences: FiniteDifferences, jvp
using LinearAlgebra: dot

DI.supports_mutation(::AutoFiniteDifferences) = DI.MutationNotSupported()

function FiniteDifferences.to_vec(a::OneElement)  # TODO: remove type piracy (https://github.com/JuliaDiff/FiniteDifferences.jl/issues/141)
    return FiniteDifferences.to_vec(collect(a))
end

function DI.value_and_pushforward(
    f::F, backend::AutoFiniteDifferences{fdm}, x, dx, extras::Nothing
) where {F,fdm}
    y = f(x)
    return y, jvp(backend.fdm, f, (x, dx))
end

function DI.value_and_pushforward!!(
    f::F, dy, backend::AutoFiniteDifferences, x, dx, extras
) where {F}
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, myupdate!!(dy, new_dy)
end

end
