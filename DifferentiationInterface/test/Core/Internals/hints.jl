using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using Test

@testset "Missing backend" begin
    e = nothing
    try
        gradient(sum, AutoZygote(), [1.0])
    catch e
    end
    msg = sprint(showerror, e)
    @test occursin("import Zygote", msg)
end
