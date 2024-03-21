using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest

using Chairmarks: Chairmarks
using DataFrames: DataFrames
using JET: JET
using Test

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

test_operators(
    [AutoZeroForward(), AutoZeroReverse()]; second_order=false, correctness=false
);

test_operators(
    [
        AutoZeroForward(),
        AutoZeroReverse(),
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ];
    first_order=false,
    correctness=false,
);

# call count (experimental)

test_operators(
    AutoZeroForward();
    correctness=false,
    type_stability=false,
    call_count=true,
    second_order=false,
    excluded=[:gradient_allocating],
);

test_operators(
    AutoZeroReverse();
    correctness=false,
    type_stability=false,
    call_count=true,
    second_order=true,
    excluded=[:multiderivative_allocating],
);

test_operators(
    [AutoZeroReverse(), SecondOrder(AutoZeroReverse(), AutoZeroForward())];
    correctness=false,
    type_stability=false,
    call_count=true,
    first_order=false,
);

test_operators(
    [SecondOrder(AutoZeroForward(), AutoZeroReverse())];
    correctness=false,
    type_stability=false,
    call_count=true,
    first_order=false,
    excluded=[:hessian_allocating],  # still quadratic
);

# allocs (experimental)

test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    allocations=true,
    second_order=false,
);

data = test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    benchmark=true,
);

df = DataFrames.DataFrame(pairs(data)...)
