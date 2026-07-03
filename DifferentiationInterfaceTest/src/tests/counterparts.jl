const FORWARD_CAPABLE_MODES = Union{ForwardMode, ForwardOrReverseMode, SymbolicMode}
const REVERSE_CAPABLE_MODES = Union{ReverseMode, ForwardOrReverseMode, SymbolicMode}

"""
    test_counterparts(backend)

Test that `forward_counterpart` and `reverse_counterpart` behave sensibly for `backend`:
the returned backends are available, they support the requested mode (unless `backend` is
returned unchanged for lack of a known counterpart), and applying the same counterpart
again leaves the mode unchanged.
"""
function test_counterparts(backend::AbstractADType)
    @testset "Counterparts: $(typeof(backend))" begin
        fc = forward_counterpart(backend)
        rc = reverse_counterpart(backend)
        @test check_available(fc)
        @test check_available(rc)
        # the counterpart must support the requested mode, except when the backend was
        # returned unchanged for lack of a known counterpart
        if fc !== backend || mode(backend) isa FORWARD_CAPABLE_MODES
            @test mode(fc) isa FORWARD_CAPABLE_MODES
        end
        if rc !== backend || mode(backend) isa REVERSE_CAPABLE_MODES
            @test mode(rc) isa REVERSE_CAPABLE_MODES
        end
        # applying the counterpart a second time leaves the mode unchanged
        @test mode(forward_counterpart(fc)) == mode(fc)
        @test mode(reverse_counterpart(rc)) == mode(rc)
    end
    return nothing
end
