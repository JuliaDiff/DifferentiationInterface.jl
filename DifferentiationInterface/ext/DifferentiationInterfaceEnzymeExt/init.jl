const HINT_END = "\n\nThis hint appears because DifferentiationInterface and Enzyme are both loaded. It does not necessarily imply that Enzyme is being called through DifferentiationInterface.\n\n"

function HINT_START(option)
    return "\nIf you are using Enzyme by selecting the `AutoEnzyme` object from ADTypes, you may want to try setting the `$option` option as follows:"
end

function __init__()
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
                elseif occursin("EnzymeRuntimeActivityError", string(nameof(T)))
                    printstyled(io, HINT_START("mode"); bold=true)
                    printstyled(
                        io,
                        "\n\n\tAutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward))\n\tAutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))";
                        color=:cyan,
                        bold=true,
                    )
                    printstyled(io, HINT_END; italic=true)
                end
            end
        end
    end
end
