## Direct

function ADTypes.jacobian_sparsity(f, x, detector::DI.DenseSparsityDetector{:direct})
    (; backend, atol) = detector
    J = jacobian(f, backend, x)
    return sparse(abs.(J) .> atol)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::DI.DenseSparsityDetector{:direct})
    (; backend, atol) = detector
    J = jacobian(f!, y, backend, x)
    return sparse(abs.(J) .> atol)
end

function ADTypes.hessian_sparsity(f, x, detector::DI.DenseSparsityDetector{:direct})
    (; backend, atol) = detector
    H = hessian(f, backend, x)
    return sparse(abs.(H) .> atol)
end

## Iterative

function ADTypes.jacobian_sparsity(f, x, detector::DI.DenseSparsityDetector{:iterative})
    (; backend, atol) = detector
    y = f(x)
    n, m = length(x), length(y)
    I, J = Int[], Int[]
    if DI.pushforward_performance(backend) isa DI.PushforwardFast
        p = similar(y)
        prep = DI.prepare_pushforward_same_point(
            f, backend, x, (DI.basis(x, first(eachindex(x))),); strict = Val(true)
        )
        for (kj, j) in enumerate(eachindex(x))
            pushforward!(f, (p,), prep, backend, x, (DI.basis(x, j),))
            for ki in LinearIndices(p)
                if abs(p[ki]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    else
        p = similar(x)
        prep = DI.prepare_pullback_same_point(
            f, backend, x, (DI.basis(y, first(eachindex(y))),); strict = Val(true)
        )
        for (ki, i) in enumerate(eachindex(y))
            pullback!(f, (p,), prep, backend, x, (DI.basis(y, i),))
            for kj in LinearIndices(p)
                if abs(p[kj]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::DI.DenseSparsityDetector{:iterative})
    (; backend, atol) = detector
    n, m = length(x), length(y)
    I, J = Int[], Int[]
    if DI.pushforward_performance(backend) isa DI.PushforwardFast
        p = similar(y)
        prep = DI.prepare_pushforward_same_point(
            f!, y, backend, x, (DI.basis(x, first(eachindex(x))),); strict = Val(true)
        )
        for (kj, j) in enumerate(eachindex(x))
            pushforward!(f!, y, (p,), prep, backend, x, (DI.basis(x, j),))
            for ki in LinearIndices(p)
                if abs(p[ki]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    else
        p = similar(x)
        prep = DI.prepare_pullback_same_point(
            f!, y, backend, x, (DI.basis(y, first(eachindex(y))),); strict = Val(true)
        )
        for (ki, i) in enumerate(eachindex(y))
            pullback!(f!, y, (p,), prep, backend, x, (DI.basis(y, i),))
            for kj in LinearIndices(p)
                if abs(p[kj]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.hessian_sparsity(f, x, detector::DI.DenseSparsityDetector{:iterative})
    (; backend, atol) = detector
    n = length(x)
    I, J = Int[], Int[]
    p = similar(x)
    prep = DI.prepare_hvp_same_point(
        f, backend, x, (DI.basis(x, first(eachindex(x))),); strict = Val(true)
    )
    for (kj, j) in enumerate(eachindex(x))
        hvp!(f, (p,), prep, backend, x, (DI.basis(x, j),))
        for ki in LinearIndices(p)
            if abs(p[ki]) > atol
                push!(I, ki)
                push!(J, kj)
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), n, n)
end
