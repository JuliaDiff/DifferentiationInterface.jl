module DifferentiationInterfaceTestLuxExt

using ComponentArrays: ComponentArray
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using Lux:
    Lux,
    BatchNorm,
    Bilinear,
    Chain,
    Conv,
    ConvTranspose,
    Dense,
    GroupNorm,
    GRUCell,
    InstanceNorm,
    LayerNorm,
    LSTMCell,
    Maxout,
    MaxPool,
    MeanPool,
    RNNCell,
    SamePad,
    Scale,
    SkipConnection,
    StatefulRecurrentCell,
    gelu,
    relu
using LuxTestUtils: check_approx
using Random: AbstractRNG, default_rng

#=
Relevant discussions:

- https://github.com/LuxDL/Lux.jl/issues/769
=#

function DIT.lux_isapprox(a, b; atol, rtol)
    return check_approx(a, b; atol, rtol)
end

function square_loss(ps, model, x, st)
    return sum(abs2, first(model(x, ps, st)))
end

function DIT.lux_scenarios(rng::AbstractRNG = default_rng())
    models_and_xs = [
        #! format: off
        (
            Dense(2, 4),
            randn(rng, Float32, 2, 3)
        ),
        (
            Dense(2, 4, gelu),
            randn(rng, Float32, 2, 3)
        ),
        (
            Dense(2, 4, gelu; use_bias=false),
            randn(rng, Float32, 2, 3)
        ),
        (
            Chain(Dense(2, 4, relu), Dense(4, 3)),
            randn(rng, Float32, 2, 3)
        ),
        (
            Scale(2),
            randn(rng, Float32, 2, 3)
        ),
        (
            Conv((3, 3), 2 => 3),
            randn(rng, Float32, 3, 3, 2, 2)
        ),
        (
            Conv((3, 3), 2 => 3, gelu; pad=SamePad()),
            randn(rng, Float32, 3, 3, 2, 2)
        ),
        (
            Conv((3, 3), 2 => 3, relu; use_bias=false, pad=SamePad()),
            randn(rng, Float32, 3, 3, 2, 2),
        ),
        (
            Chain(Conv((3, 3), 2 => 3, gelu), Conv((3, 3), 3 => 1, gelu)),
            rand(rng, Float32, 5, 5, 2, 2),
        ),
        (
            Chain(Conv((4, 4), 2 => 2; pad=SamePad()), MeanPool((5, 5); pad=SamePad())),
            rand(rng, Float32, 5, 5, 2, 2),
        ),
        (
            Chain(Conv((3, 3), 2 => 3, relu; pad=SamePad()), MaxPool((2, 2))),
            rand(rng, Float32, 5, 5, 2, 2),
        ),
        (
            Maxout(() -> Dense(5 => 4, tanh), 3),
            randn(rng, Float32, 5, 2)
        ),
        (
            Bilinear((2, 2) => 3),
            randn(rng, Float32, 2, 3)
        ),
        (
            SkipConnection(Dense(2 => 2), vcat),
            randn(rng, Float32, 2, 3)
        ),
        (
            ConvTranspose((3, 3), 3 => 2; stride=2),
            rand(rng, Float32, 5, 5, 3, 1)
        ),
        (
            StatefulRecurrentCell(RNNCell(3 => 5)),
            rand(rng, Float32, 3, 2)
        ),
        (
            StatefulRecurrentCell(RNNCell(3 => 5, gelu)),
            rand(rng, Float32, 3, 2)
        ),
        (
            StatefulRecurrentCell(RNNCell(3 => 5, gelu; use_bias=false)),
            rand(rng, Float32, 3, 2),
        ),
        (
            Chain(StatefulRecurrentCell(RNNCell(3 => 5)), StatefulRecurrentCell(RNNCell(5 => 3)),),
            rand(rng, Float32, 3, 2),
        ),
        (
            StatefulRecurrentCell(LSTMCell(3 => 5)),
            rand(rng, Float32, 3, 2)
        ),
        (
            Chain(StatefulRecurrentCell(LSTMCell(3 => 5)), StatefulRecurrentCell(LSTMCell(5 => 3)),),
            rand(rng, Float32, 3, 2),
        ),
        (
            StatefulRecurrentCell(GRUCell(3 => 5)),
            rand(rng, Float32, 3, 10)
        ),
        (
            Chain(StatefulRecurrentCell(GRUCell(3 => 5)), StatefulRecurrentCell(GRUCell(5 => 3)),),
            rand(rng, Float32, 3, 10),
        ),
        (
            Chain(Dense(2, 4), BatchNorm(4)),
            randn(rng, Float32, 2, 3)
        ),
        (
            Chain(Dense(2, 4), BatchNorm(4, gelu)),
            randn(rng, Float32, 2, 3)
        ),
        (
            Chain(Dense(2, 4), BatchNorm(4, gelu; track_stats=false)),
            randn(rng, Float32, 2, 3),
        ),
        (
            Chain(Conv((3, 3), 2 => 6), BatchNorm(6)),
            randn(rng, Float32, 6, 6, 2, 2)
        ),
        (
            Chain(Conv((3, 3), 2 => 6, tanh), BatchNorm(6)),
            randn(rng, Float32, 6, 6, 2, 2)
        ),
        (
            Chain(Dense(2, 4), GroupNorm(4, 2, gelu)),
            randn(rng, Float32, 2, 3)
        ),
        (
            Chain(Dense(2, 4), GroupNorm(4, 2)),
            randn(rng, Float32, 2, 3)
        ),
        (
            Chain(Conv((3, 3), 2 => 6), GroupNorm(6, 3)),
            randn(rng, Float32, 6, 6, 2, 2)
        ),
        (
            Chain(Conv((3, 3), 2 => 6, tanh), GroupNorm(6, 3)),
            randn(rng, Float32, 6, 6, 2, 2),
        ),
        (
            Chain(Conv((3, 3), 2 => 3, gelu), LayerNorm((1, 1, 3))),
            randn(rng, Float32, 4, 4, 2, 2),
        ),
        (
            Chain(Conv((3, 3), 2 => 6), InstanceNorm(6)),
            randn(rng, Float32, 6, 6, 2, 2)
        ),
        (
            Chain(Conv((3, 3), 2 => 6, tanh), InstanceNorm(6)),
            randn(rng, Float32, 6, 6, 2, 2),
        ),
        #! format: on
    ]

    scens = DIT.Scenario[]

    for (model, x) in models_and_xs
        ps, st = Lux.setup(rng, model)
        g = DI.gradient(
            ps -> square_loss(ps, model, x, st), DI.AutoForwardDiff(), ComponentArray(ps)
        )
        scen = DIT.Scenario{:gradient, :out}(
            square_loss,
            ComponentArray(ps),
            DI.Constant(model),
            DI.Constant(x),
            DI.Constant(st);
            prep_args = (
                x = ComponentArray(ps),
                contexts = (DI.Constant(model), DI.Constant(x), DI.Constant(st)),
            ),
            res1 = g,
        )
        push!(scens, scen)
    end

    return scens
end

end
