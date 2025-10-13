"""
    check_available(backend)

Check whether `backend` is available (i.e. whether the extension is loaded) and return a `Bool`.
"""
check_available(backend::AbstractADType) = false

function check_available(backend::SecondOrder)
    return check_available(inner(backend)) && check_available(outer(backend))
end

check_available(backend::AutoSparse) = check_available(dense_ad(backend))

function check_available(backend::MixedMode)
    return check_available(forward_backend(backend)) &&
        check_available(reverse_backend(backend))
end

check_available(::ADTypes.NoAutoDiff) = throw(ADTypes.NoAutoDiffSelectedError())

"""
    check_inplace(backend)

Check whether `backend` supports differentiation of in-place functions and return a `Bool`.
"""
check_inplace(backend::AbstractADType) = Bool(inplace_support(backend))
