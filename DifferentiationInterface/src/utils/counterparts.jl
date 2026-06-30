"""
    forward_counterpart(backend)

Return the forward-mode counterpart of `backend`, if it exists.
"""
function forward_counterpart(backend::AbstractADType)
    mode(backend) isa ReverseMode &&
        throw(ArgumentError("No forward-mode counterpart known for `$backend`."))
    return backend
end
forward_counterpart(backend::AutoMooncake) = AutoMooncakeForward(; config = backend.config)

"""
    reverse_counterpart(backend)

Return the reverse-mode counterpart of `backend`, if it exists.
"""
function reverse_counterpart(backend::AbstractADType)
    mode(backend) isa ForwardMode &&
        throw(ArgumentError("No reverse-mode counterpart known for `$backend`."))
    return backend
end
reverse_counterpart(backend::AutoMooncakeForward) = AutoMooncake(; config = backend.config)

# AutoEnzyme counterparts need the `Forward`/`Reverse` mode objects from EnzymeCore,
# so they live in DifferentiationInterfaceEnzymeExt.
