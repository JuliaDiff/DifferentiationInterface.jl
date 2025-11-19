to_reac(x::AbstractArray) = to_rarray(x)
to_reac(x::ConcreteRArray) = x
to_reac(x::Number) = ConcreteRNumber(x)
to_reac(x::ConcreteRNumber) = x

to_reac(c::DI.Constant) = DI.Constant(to_reac(DI.unwrap(c)))
to_reac(c::DI.Cache) = DI.Cache(to_reac(DI.unwrap(c)))
