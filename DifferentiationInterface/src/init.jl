function __init__()
    return Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f in (_prepare_pushforward_aux, _prepare_pullback_aux)
            B = first(T for T in argtypes if T <: AbstractADType)
            packages = required_packages(B)
            loaded = map(string, values(Base.loaded_modules))
            missing_package = any(!(p in loaded) for p in packages)
            if missing_package
                import_statement = "import $(packages[1])"
                for p in packages[2:end]
                    import_statement *= ", $p"
                end
                printstyled(
                    io,
                    "\n\nThe autodiff backend you chose requires a package which may not be loaded. Please run the following command and try again:";
                    bold = true,
                )
                printstyled(io, "\n\n\t$import_statement"; color = :cyan, bold = true)
            else
                printstyled(
                    io,
                    "\n\nThe autodiff backend you chose may not be compatible with the operation you want to perform. Please refer to the documentation of DifferentiationInterface.jl and open an issue if necessary.";
                    bold = true,
                )
            end
        end
    end
end
