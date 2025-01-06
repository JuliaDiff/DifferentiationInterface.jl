function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f in (_prepare_pushforward_aux, _prepare_pullback_aux)
            B = first(T for T in argtypes if T <: AbstractADType)
            printstyled(
                io,
                "\n\nThe autodiff backend package you want to use may not be loaded. Please run the following command and try again:";
                bold=true,
            )
            printstyled(io, "\n\n\timport $(package_name(B))"; color=:cyan, bold=true)
        end
    end
end
