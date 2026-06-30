"""
    test_counterparts(backend)

Test that `forward_counterpart` and `reverse_counterpart` behave sensibly for `backend`:
the returned backends are available, and applying the counterpart again leaves the mode
unchanged (idempotence on the counterpart's own output).
"""
function test_counterparts(backend::AbstractADType)
    @testset "Counterparts: $(typeof(backend))" begin
        fc = forward_counterpart(backend)
        rc = reverse_counterpart(backend)
        @test check_available(fc)
        @test check_available(rc)
        @test mode(forward_counterpart(fc)) == mode(fc)
        @test mode(reverse_counterpart(rc)) == mode(rc)
    end
    return nothing
end
