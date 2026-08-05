function with_value end
function without_value end
function has_value end

function inplace end
function is_inplace end

function standard end

function order end
function preparator end
function preparator! end

function needs_tangent end

const FIRST_ORDER = [:pushforward, :pullback, :derivative, :gradient, :jacobian]
const SECOND_ORDER = [:hvp, :second_derivative, :hessian]
const ALL_OPERATORS = vcat(FIRST_ORDER, SECOND_ORDER)

for op in ALL_OPERATORS
    op! = Symbol(op, "!")
    val_prefix = if op == :second_derivative
        "value_derivative_and_"
    elseif op == :hessian
        "value_gradient_and_"
    elseif op == :hvp
        "gradient_and_"
    else
        "value_and_"
    end
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)

    prep_op = Symbol("prepare_", op)
    prep_op! = Symbol("prepare!_", op)
    prep_op_same = Symbol("prepare_", op, "_same_point")

    @eval standard(
        ::Union{
            typeof($op),
            typeof($op!),
            typeof($val_and_op),
            typeof($val_and_op!),
        }
    ) = $op

    @eval with_value(::Union{typeof($op), typeof($val_and_op)}) = $val_and_op
    @eval with_value(::Union{typeof($op!), typeof($val_and_op!)}) = $val_and_op!

    @eval without_value(::Union{typeof($op), typeof($val_and_op)}) = $op
    @eval without_value(::Union{typeof($op!), typeof($val_and_op!)}) = $op!

    @eval has_value(::Union{typeof($op), typeof($op!)}) = false
    @eval has_value(::Union{typeof($val_and_op), typeof($val_and_op!)}) = true

    @eval inplace(::Union{typeof($op), typeof($val_and_op)}) = $op
    @eval inplace(::Union{typeof($op!), typeof($val_and_op!)}) = $op!

    @eval is_inplace(::Union{typeof($op), typeof($val_and_op)}) = false
    @eval is_inplace(::Union{typeof($op!), typeof($val_and_op!)}) = true

    @eval preparator(
        ::Union{
            typeof($op),
            typeof($op!),
            typeof($val_and_op),
            typeof($val_and_op!),
        }
    ) = $prep_op

    @eval preparator!(
        ::Union{
            typeof($op),
            typeof($op!),
            typeof($val_and_op),
            typeof($val_and_op!),
        }
    ) = $prep_op!

    if op in FIRST_ORDER
        @eval order(
            ::Union{
                typeof($op),
                typeof($op!),
                typeof($val_and_op),
                typeof($val_and_op!),
            }
        ) = 1
    else
        @eval order(
            ::Union{
                typeof($op),
                typeof($op!),
                typeof($val_and_op),
                typeof($val_and_op!),
            }
        ) = 2
    end

    if op in [:pushforward, :pullback, :hvp]
        @eval needs_tangent(
            ::Union{
                typeof($op),
                typeof($op!),
                typeof($val_and_op),
                typeof($val_and_op!),
            }
        ) = true
    else
        @eval needs_tangent(
            ::Union{
                typeof($op),
                typeof($op!),
                typeof($val_and_op),
                typeof($val_and_op!),
            }
        ) = false
    end
end
