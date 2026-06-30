## Backend counterparts

# Flip the Enzyme mode while preserving its other type parameters (and the function
# annotation `A`).

function DI.forward_counterpart(
        ::AutoEnzyme{
            <:ReverseMode{
                ReturnPrimal, RuntimeActivity, StrongZero, ABI, Holomorphic, ErrIfFuncWritten,
            },
            A,
        },
    ) where {ReturnPrimal, RuntimeActivity, StrongZero, ABI, Holomorphic, ErrIfFuncWritten, A}
    mode = ForwardMode{ReturnPrimal, ABI, ErrIfFuncWritten, RuntimeActivity, StrongZero}()
    return AutoEnzyme(; mode, function_annotation = A)
end

function DI.reverse_counterpart(
        ::AutoEnzyme{
            <:ForwardMode{ReturnPrimal, ABI, ErrIfFuncWritten, RuntimeActivity, StrongZero},
            A,
        },
    ) where {ReturnPrimal, ABI, ErrIfFuncWritten, RuntimeActivity, StrongZero, A}
    mode = ReverseMode{ReturnPrimal, RuntimeActivity, StrongZero, ABI, false, ErrIfFuncWritten}()
    return AutoEnzyme(; mode, function_annotation = A)
end
