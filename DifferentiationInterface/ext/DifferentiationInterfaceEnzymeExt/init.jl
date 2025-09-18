const HINT_END = "\n\nThis hint appears because DifferentiationInterface and Enzyme are both loaded. It does not necessarily imply that Enzyme is being called through DifferentiationInterface.\n\n"

function HINT_START(option)
    return "\nIf you are using Enzyme by selecting the `AutoEnzyme` object from ADTypes, you may want to try setting the `$option` option as follows:"
end

function __init__()
    if !isdefined(Base.Experimental, :register_error_hint)
        return nothing
    end
    # robust against internal changes
    condition = (
        isdefined(Enzyme, :Compiler) &&
        Enzyme.Compiler isa Module &&
        isdefined(Enzyme.Compiler, :EnzymeError) &&
        Enzyme.Compiler.EnzymeError isa DataType
    )
    condition || return nothing
    # see https://github.com/JuliaLang/julia/issues/58367 for why this isn't easier
    for n in names(Enzyme.Compiler; all=true)
        T = getfield(Enzyme.Compiler, n)
        if T isa DataType && T <: Enzyme.Compiler.EnzymeError
            # robust against internal changes
            Base.Experimental.register_error_hint(T) do io, exc
                if occursin("EnzymeMutabilityException", string(nameof(T)))
                    printstyled(io, HINT_START("function_annotation"); bold=true)
                    printstyled(
                        io,
                        "\n\n\tAutoEnzyme(; function_annotation=Enzyme.Duplicated)";
                        color=:cyan,
                        bold=true,
                    )
                    printstyled(io, HINT_END; italic=true)
                end
                # EnzymeRuntimeActivityError is no longer a concrete type since https://github.com/EnzymeAD/Enzyme.jl/pull/2555 (now a UnionAll) so we cannot define a hint
            end
        end
    end
end
