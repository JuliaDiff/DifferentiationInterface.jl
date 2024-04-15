struct SparseDiffToolsTwoArgJacobianExtras{C} <: JacobianExtras
    cache::C
end

for AutoSparse in SPARSE_BACKENDS
    @eval begin
        ## Jacobian

        function DI.prepare_jacobian(
            f!, y::AbstractArray, backend::$AutoSparse, x::AbstractArray
        )
            cache = sparse_jacobian_cache(
                backend, SymbolicsSparsityDetection(), f!, similar(y), x
            )
            return SparseDiffToolsTwoArgJacobianExtras(cache)
        end

        function DI.value_and_jacobian(
            f!, y, backend::$AutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
        )
            jac = sparse_jacobian(backend, extras.cache, f!, y, x)
            f!(y, x)
            return y, jac
        end

        function DI.value_and_jacobian!(
            f!, y, jac, backend::$AutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
        )
            sparse_jacobian!(jac, backend, extras.cache, f!, y, x)
            f!(y, x)
            return y, jac
        end

        function DI.jacobian(
            f!, y, backend::$AutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
        )
            jac = sparse_jacobian(backend, extras.cache, f!, y, x)
            return jac
        end

        function DI.jacobian!(
            f!, y, jac, backend::$AutoSparse, x, extras::SparseDiffToolsTwoArgJacobianExtras
        )
            sparse_jacobian!(jac, backend, extras.cache, f!, y, x)
            return jac
        end
    end
end
