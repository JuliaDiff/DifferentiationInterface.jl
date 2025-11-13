get_config(::AnyAutoMooncake{Nothing}) = Config()
get_config(backend::AnyAutoMooncake{<:Config}) = backend.config

@inline zero_tangent_unwrap(c::DI.Context) = zero_tangent(DI.unwrap(c))
@inline Dual_unwrap(c, dc) = Dual(DI.unwrap(c), dc)
@inline CoDual_unwrap(c, dc) = CoDual(DI.unwrap(c), dc)
