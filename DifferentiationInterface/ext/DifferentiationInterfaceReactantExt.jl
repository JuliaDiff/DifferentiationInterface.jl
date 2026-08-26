module DifferentiationInterfaceReactantExt

using ADTypes: AutoEnzyme
import DifferentiationInterface as DI
using Reactant: within_compile

DI._use_reactant_jacobian(::AutoEnzyme) = within_compile()

end
