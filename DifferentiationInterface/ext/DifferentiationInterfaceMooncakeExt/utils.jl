get_config(::AnyAutoMooncake{Nothing}) = Config()
get_config(backend::AnyAutoMooncake{<:Config}) = backend.config

@inline zero_tangent_unwrap(c::DI.Context) = zero_tangent(DI.unwrap(c))
@inline first_unwrap(c, dc) = (DI.unwrap(c), dc)
