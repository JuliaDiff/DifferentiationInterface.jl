_to_reactant(x) = DI.to_reactant(x)
_to_reactant(c::DI.Constant) = DI.Constant(_to_reactant(DI.unwrap(c)))
_to_reactant(c::DI.Cache) = DI.Cache(_to_reactant(DI.unwrap(c)))
