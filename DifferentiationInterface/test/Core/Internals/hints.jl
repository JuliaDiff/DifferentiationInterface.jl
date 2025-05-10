using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using Test

@testset "Missing backend" begin
    msg = try
        gradient(sum, AutoZygote(), [1.0])
    catch e
        buf = IOBuffer()
        showerror(buf, e)
        String(take!(buf))
    end
    @test occursin("import Zygote", msg)
end
