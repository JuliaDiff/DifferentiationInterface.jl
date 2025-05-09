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
                    printstyled(
                        io,
                        "\nIf you are using Enzyme through DifferentiationInterface, you may want to try modifying the ADTypes backend object as follows:";
                        bold=true,
                    )
                    printstyled(
                        io,
                        "\n\n\tAutoEnzyme(; function_annotation=Enzyme.Duplicated)\n\n";
                        color=:cyan,
                        bold=true,
                    )
                elseif occursin("EnzymeRuntimeActivityError", string(nameof(T)))
                    printstyled(
                        io,
                        "\nIf you are using Enzyme through DifferentiationInterface, you may want to try modifying the ADTypes backend object as follows:";
                        bold=true,
                    )
                    printstyled(
                        io,
                        "\n\n\tAutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward))\n\tAutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))\n\n";
                        color=:cyan,
                        bold=true,
                    )
                end
            end
        end
    end
end
