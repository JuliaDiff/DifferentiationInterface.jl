## Vector to vector

diffsquare(x::AbstractVector)::AbstractVector = diff(x) .^ 2
diffcube(x::AbstractVector)::AbstractVector = diff(x) .^ 3

function diffsquare!(y::AbstractVector, x::AbstractVector)
    x1 = @view x[1:(end - 1)]
    x2 = @view x[2:end]
    y .= x2 .- x1
    y .^= 2
    return nothing
end

function diffcube!(y::AbstractVector, x::AbstractVector)
    x1 = @view x[1:(end - 1)]
    x2 = @view x[2:end]
    y .= x2 .- x1
    y .^= 3
    return nothing
end

function diffsquare_jacobian(x)
    n = length(x)
    return spdiagm(n - 1, n, 0 => -2 * diff(x), 1 => 2 * diff(x))
end

function diffcube_jacobian(x)
    n = length(x)
    return spdiagm(n - 1, n, 0 => -3 * diff(x) .^ 2, 1 => 3 * diff(x) .^ 2)
end

function sparse_vec_to_vec_scenarios(x::AbstractVector)
    f = diffsquare
    f! = diffsquare!
    y = f(x)
    jac = diffsquare_jacobian(x)

    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                Scenario{:jacobian,pl_op}(f, x; res1=jac),
                Scenario{:jacobian,pl_op}(f!, y, x; res1=jac),
            ],
        )
    end
    return scens
end

## Matrix to vector

function diffsquarecube_matvec(x::AbstractMatrix)::AbstractVector
    return vcat(diffsquare(vec(x)), diffcube(vec(x)))
end

function diffsquarecube_matvec!(y::AbstractVector, x::AbstractMatrix)
    m, n = size(x)
    diffsquare!(view(y, 1:(m * n - 1)), vec(x))
    diffcube!(view(y, (m * n):(2(m * n) - 2)), vec(x))
    return nothing
end

function diffsquarecube_matvec_jacobian(x)
    return vcat(diffsquare_jacobian(vec(x)), diffcube_jacobian(vec(x)))
end

function sparse_mat_to_vec_scenarios(x::AbstractMatrix)
    f = diffsquarecube_matvec
    f! = diffsquarecube_matvec!
    y = f(x)
    jac = diffsquarecube_matvec_jacobian(x)

    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                Scenario{:jacobian,pl_op}(f, x; res1=jac),
                Scenario{:jacobian,pl_op}(f!, y, x; res1=jac),
            ],
        )
    end
    return scens
end

## Vector to matrix

diffsquarecube_vecmat(x::AbstractVector)::AbstractMatrix = hcat(diffsquare(x), diffcube(x))

function diffsquarecube_vecmat!(y::AbstractMatrix, x::AbstractVector)
    diffsquare!(view(y, :, 1), x)
    diffcube!(view(y, :, 2), x)
    return nothing
end

function diffsquarecube_vecmat_jacobian(x::AbstractVector)
    return vcat(diffsquare_jacobian(x), diffcube_jacobian(x))
end

function sparse_vec_to_mat_scenarios(x::AbstractVector)
    f = diffsquarecube_vecmat
    f! = diffsquarecube_vecmat!
    y = f(x)
    jac = diffsquarecube_vecmat_jacobian(vec(x))

    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                Scenario{:jacobian,pl_op}(f, x; res1=jac),
                Scenario{:jacobian,pl_op}(f!, y, x; res1=jac),
            ],
        )
    end
    return scens
end

## Matrix to matrix

function diffsquarecube_matmat(x::AbstractMatrix)::AbstractMatrix
    return hcat(diffsquare(vec(x)), diffcube(vec(x)))
end

function diffsquarecube_matmat!(y::AbstractMatrix, x::AbstractMatrix)
    diffsquare!(view(y, :, 1), vec(x))
    diffcube!(view(y, :, 2), vec(x))
    return nothing
end

function diffsquarecube_matmat_jacobian(x::AbstractMatrix)
    return vcat(diffsquare_jacobian(vec(x)), diffcube_jacobian(vec(x)))
end

function sparse_mat_to_mat_scenarios(x::AbstractMatrix)
    f = diffsquarecube_matmat
    f! = diffsquarecube_matmat!
    y = f(x)
    jac = diffsquarecube_matmat_jacobian(x)

    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(
            scens,
            [
                Scenario{:jacobian,pl_op}(f, x; res1=jac),
                Scenario{:jacobian,pl_op}(f!, y, x; res1=jac),
            ],
        )
    end
    return scens
end

## Vector to scalar

sumdiffcube(x::AbstractVector)::Number = sum(diffcube(x))

function sumdiffcube_gradient(x::AbstractVector)
    g = similar(x)
    for j in eachindex(x)
        if j == firstindex(x)
            g[j] = -3(x[j + 1] - x[j])^2
        elseif j == lastindex(x)
            g[j] = +3(x[j] - x[j - 1])^2
        else
            g[j] = 3(x[j] - x[j - 1])^2 - 3(x[j + 1] - x[j])^2
        end
    end
    return g
end

function sumdiffcube_hessian(x::AbstractVector)
    T = eltype(x)
    d = 6 * diff(x)
    return spdiagm(0 => vcat(d, zero(T)) + vcat(zero(T), d), 1 => -d, -1 => -d)
end

function sparse_vec_to_num_scenarios(x::AbstractVector)
    f = sumdiffcube
    grad = sumdiffcube_gradient(x)
    hess = sumdiffcube_hessian(x)

    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(scens, [Scenario{:hessian,pl_op}(f, x; res1=grad, res2=hess)])
    end
    return scens
end

## Matrix to scalar

sumdiffcube_mat(x::AbstractMatrix)::Number = sum(diffcube(vec(x)))

sumdiffcube_mat_gradient(x::AbstractMatrix) = reshape(sumdiffcube_gradient(vec(x)), size(x))

function sumdiffcube_mat_hessian(x::AbstractMatrix)
    T = eltype(x)
    d = 6 * diff(vec(x))
    return spdiagm(0 => vcat(d, zero(T)) + vcat(zero(T), d), 1 => -d, -1 => -d)
end

function sparse_mat_to_num_scenarios(x::AbstractMatrix)
    f = sumdiffcube_mat
    grad = sumdiffcube_mat_gradient(x)
    hess = sumdiffcube_mat_hessian(x)

    scens = Scenario[]
    for pl_op in (:out, :in)
        append!(scens, [Scenario{:hessian,pl_op}(f, x; res1=grad, res2=hess)])
    end
    return scens
end

## Various matrices

function banded_matrix(::Type{T}, n, b) where {T}
    @assert b <= n
    pairs = [k => rand(T, n - k) for k in 0:b]
    return spdiagm(n, n, pairs...)
end

### Linear map

struct SquareLinearMap{M<:AbstractMatrix}
    A::M
end

function Base.show(io::IO, s::SquareLinearMap{M}) where {M}
    return print(io, "SquareLinearMap{$M - $(size(s.A)) with $(mynnz(s.A)) nonzeros}")
end

function (s::SquareLinearMap)(x::AbstractArray)
    return s.A * abs2.(vec(x))
end

function (s::SquareLinearMap)(y::AbstractArray, x::AbstractArray)
    vec(y) .= s.A * abs2.(vec(x))
    return nothing
end

function squarelinearmap_jacobian(x::AbstractArray, A::AbstractMatrix)
    return 2 .* A .* transpose(vec(x))
end

function squarelinearmap_scenarios(x::AbstractVector, band_sizes)
    n = length(x)
    scens = Scenario[]
    for A in banded_matrix.(eltype(x), n, band_sizes)
        f = SquareLinearMap(A)
        f! = f
        y = f(x)
        jac = sparse(squarelinearmap_jacobian(x, A))
        for pl_op in (:out, :in)
            append!(
                scens,
                [
                    Scenario{:jacobian,pl_op}(f, x; res1=jac),
                    Scenario{:jacobian,pl_op}(f!, y, x; res1=jac),
                ],
            )
        end
    end
    return scens
end

### Quadratic form

struct SquareQuadraticForm{M<:AbstractMatrix}
    A::M
end

function Base.show(io::IO, s::SquareQuadraticForm{M}) where {M}
    return print(io, "SquareQuadraticForm{$M - $(size(s.A)) with $(mynnz(s.A)) nonzeros}")
end

function (s::SquareQuadraticForm)(x::AbstractArray)
    v = abs2.(vec(x))
    return dot(v, s.A, v)
end

function squarequadraticform_gradient(x::AbstractArray, A::AbstractMatrix)
    g = similar(x)
    for i in eachindex(g)
        g[i] =
            4 * A[i, i] * x[i]^3 +
            2 * sum((A[i, j] + A[j, i]) * x[i] * x[j]^2 for j in eachindex(g) if j != i)
    end
    return g
end

function squarequadraticform_hessian(x::AbstractArray, A::AbstractMatrix)
    H = similar(x, length(x), length(x))
    for i in axes(H, 1), j in axes(H, 2)
        if i == j
            H[i, i] =
                12 * A[i, i] * x[i]^2 +
                2 * sum((A[i, j2] + A[j2, i]) * x[j2]^2 for j2 in axes(H, 2) if j2 != i)
        else
            H[i, j] = 4 * (A[i, j] + A[j, i]) * x[i] * x[j]
        end
    end
    return H
end

function squarequadraticform_scenarios(x::AbstractVector, band_sizes)
    n = length(x)
    scens = Scenario[]
    for A in banded_matrix.(eltype(x), n, band_sizes)
        f = SquareQuadraticForm(A)
        grad = squarequadraticform_gradient(x, A)
        hess = sparse(squarequadraticform_hessian(x, A))
        for pl_op in (:out, :in)
            push!(scens, Scenario{:hessian,pl_op}(f, x; res1=grad, res2=hess))
        end
    end
    return scens
end

## Gather

"""
    sparse_scenarios()

Create a vector of [`Scenario`](@ref)s with sparse array types, focused on sparse Jacobians and Hessians.
"""
function sparse_scenarios(;
    band_sizes=[5, 10, 20], include_constantified=false, include_cachified=false
)
    x_6 = float.(1:6)
    x_2_3 = float.(reshape(1:6, 2, 3))
    x_50 = float.(range(1, 2, 50))

    scens = vcat(
        sparse_vec_to_vec_scenarios(x_6),
        sparse_vec_to_mat_scenarios(x_6),
        sparse_mat_to_vec_scenarios(x_2_3),
        sparse_mat_to_mat_scenarios(x_2_3),
        sparse_vec_to_num_scenarios(x_6),
        sparse_mat_to_num_scenarios(x_2_3),
    )
    if !isempty(band_sizes)
        append!(scens, squarelinearmap_scenarios(x_50, band_sizes))
        append!(scens, squarequadraticform_scenarios(x_50, band_sizes))
    end

    final_scens = Scenario[]
    append!(final_scens, scens)
    include_constantified && append!(final_scens, constantify(scens))
    include_cachified && append!(final_scens, cachify(scens))
    return final_scens
end
