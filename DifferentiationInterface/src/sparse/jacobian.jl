struct SparseJacobianExtras{
    args,dir,C<:CompressedMatrix{dir},S<:AbstractVector,P<:AbstractVector,E<:Extras
} <: JacobianExtras
    compressed::C
    seeds::S
    products::P
    jp_extras::E
end

function SparseJacobianExtras{args}(;
    compressed::CompressedMatrix{dir}, seeds::S, products::P, jp_extras::E
) where {args,dir,S,P,E}
    if dir == :col
        @assert jp_extras isa PushforwardExtras
    elseif dir == :row
        @assert jp_extras isa PullbackExtras
    end
    C = typeof(compressed)
    return SparseJacobianExtras{args,dir,C,S,P,E}(compressed, seeds, products, jp_extras)
end

## Jacobian, one argument

function prepare_jacobian(f, backend::AutoSparse, x)
    y = f(x)
    sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    if Bool(pushforward_performance(backend))
        colors = column_coloring(sparsity, coloring_algorithm(backend))
        groups = get_groups(colors)
        seeds = map(groups) do group
            seed = zero(x)
            seed[group] .= one(eltype(x))
            seed
        end
        jp_extras = prepare_pushforward(f, backend, x, first(seeds))
        products = map(seeds) do seed
            pushforward(f, backend, x, seed, jp_extras)
        end
        aggregates = stack(vec, products; dims=2)
        compressed = CompressedMatrix{:col}(sparsity, colors, groups, aggregates)
    else
        colors = row_coloring(sparsity, coloring_algorithm(backend))
        groups = get_groups(colors)
        seeds = map(groups) do group
            seed = zero(y)
            seed[group] .= one(eltype(y))
            seed
        end
        jp_extras = prepare_pullback(f, backend, x, first(seeds))
        products = map(seeds) do seed
            pullback(f, backend, x, seed, jp_extras)
        end
        aggregates = stack(vec, products; dims=1)
        compressed = CompressedMatrix{:row}(sparsity, colors, groups, aggregates)
    end
    return SparseJacobianExtras{1}(; compressed, seeds, products, jp_extras)
end

function jacobian!(f, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{1,:col})
    (; compressed, seeds, products, jp_extras) = extras
    for k in eachindex(seeds, products)
        pushforward!(f, products[k], backend, x, seeds[k], jp_extras)
        copyto!(view(compressed.aggregates, :, k), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian!(f, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{1,:row})
    (; compressed, seeds, products, jp_extras) = extras
    for k in eachindex(seeds, products)
        pullback!(f, products[k], backend, x, seeds[k], jp_extras)
        copyto!(view(compressed.aggregates, k, :), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian(f, backend::AutoSparse, x, extras::SparseJacobianExtras{1})
    jac = similar(extras.compressed.sparsity, eltype(x))
    return jacobian!(f, jac, backend, x, extras)
end

function value_and_jacobian!(
    f, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{1}
)
    return f(x), jacobian!(f, jac, backend, x, extras)
end

function value_and_jacobian(f, backend::AutoSparse, x, extras::SparseJacobianExtras{1})
    return f(x), jacobian(f, backend, x, extras)
end

## Jacobian, two arguments

function prepare_jacobian(f!, y, backend::AutoSparse, x)
    sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    if Bool(pushforward_performance(backend))
        colors = column_coloring(sparsity, coloring_algorithm(backend))
        groups = get_groups(colors)
        seeds = map(groups) do group
            seed = zero(x)
            seed[group] .= one(eltype(x))
            seed
        end
        jp_extras = prepare_pushforward(f!, y, backend, x, first(seeds))
        products = map(seeds) do seed
            pushforward(f!, y, backend, x, seed, jp_extras)
        end
        aggregates = stack(vec, products; dims=2)
        compressed = CompressedMatrix{:col}(sparsity, colors, groups, aggregates)
    else
        colors = row_coloring(sparsity, coloring_algorithm(backend))
        groups = get_groups(colors)
        seeds = map(groups) do group
            seed = zero(y)
            seed[group] .= one(eltype(y))
            seed
        end
        jp_extras = prepare_pullback(f!, y, backend, x, first(seeds))
        products = map(seeds) do seed
            pullback(f!, y, backend, x, seed, jp_extras)
        end
        aggregates = stack(vec, products; dims=1)
        compressed = CompressedMatrix{:row}(sparsity, colors, groups, aggregates)
    end
    return SparseJacobianExtras{2}(; compressed, seeds, products, jp_extras)
end

function jacobian!(f!, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{2,:col})
    (; compressed, seeds, products, jp_extras) = extras
    for k in eachindex(seeds, products)
        pushforward!(f!, y, products[k], backend, x, seeds[k], jp_extras)
        copyto!(view(compressed.aggregates, :, k), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian!(f!, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{2,:row})
    (; compressed, seeds, products, jp_extras) = extras
    for k in eachindex(seeds, products)
        pullback!(f!, y, products[k], backend, x, seeds[k], jp_extras)
        copyto!(view(compressed.aggregates, k, :), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian(f!, y, backend::AutoSparse, x, extras::SparseJacobianExtras{2})
    jac = similar(extras.compressed.sparsity, eltype(x))
    return jacobian!(f!, y, jac, backend, x, extras)
end

function value_and_jacobian!(
    f!, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{2}
)
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

function value_and_jacobian(f!, y, backend::AutoSparse, x, extras::SparseJacobianExtras{2})
    jac = jacobian(f!, y, backend, x, extras)
    f!(y, x)
    return y, jac
end
