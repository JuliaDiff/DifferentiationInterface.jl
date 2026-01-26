struct CallCounter{F}
    f::F
    count::Base.RefValue{Int}
end

CallCounter(f::F) where {F} = CallCounter{F}(f, Ref(0))

function (cc::CallCounter)(x, args...)
    cc.count[] += 1
    return cc.f(x, args...)
end

function (cc::CallCounter)(y, x, args...)
    cc.count[] += 1
    return cc.f(y, x, args...)
end

function reset_count!(cc::CallCounter)
    count = cc.count[]
    cc.count[] = 0
    return count
end

@kwdef struct CallsResult
    preparation::Int = -1
    prepared_valop::Int = -1
    prepared_op::Int = -1
    unprepared_valop::Int = -1
    unprepared_op::Int = -1
end

"""
    DifferentiationBenchmarkDataRow

Ad-hoc storage type for differentiation benchmarking results.

# Fields

$(TYPEDFIELDS)

See the documentation of [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl) for more details on the measurement fields.
"""
Base.@kwdef struct DifferentiationBenchmarkDataRow{T} <: AbstractRow
    "backend used for benchmarking"
    backend::AbstractADType
    "scenario used for benchmarking"
    scenario::Scenario
    "differentiation operator used for benchmarking, e.g. `:gradient` or `:hessian`"
    operator::Symbol
    "whether the operator had been prepared"
    prepared::Union{Nothing, Bool}
    "number of calls to the differentiated function for one call to the operator"
    calls::Int
    "number of benchmarking samples taken"
    samples::Int
    "number of evaluations used for averaging in each sample"
    evals::Int
    "aggregated runtime over all samples, in seconds"
    time::T
    "aggregated number of allocations over all samples"
    allocs::T
    "aggregated memory allocated over all samples, in bytes"
    bytes::T
    "aggregated fraction of time spent in garbage collection over all samples, between 0.0 and 1.0"
    gc_fraction::T
    "aggregated fraction of time spent compiling over all samples, between 0.0 and 1.0"
    compile_fraction::T
end

Tables.getcolumn(row::DifferentiationBenchmarkDataRow, i::Int) = getfield(row, i)
Tables.getcolumn(row::DifferentiationBenchmarkDataRow, nm::Symbol) = getfield(row, nm)
Tables.columnnames(row::DifferentiationBenchmarkDataRow) = fieldnames(typeof(row))

"""
    DifferentiationBenchmark

# Fields

$(TYPEDFIELDS)
"""
struct DifferentiationBenchmark{T}
    rows::Vector{DifferentiationBenchmarkDataRow{T}}
end

function DifferentiationBenchmark()
    return DifferentiationBenchmark(DifferentiationBenchmarkDataRow{Float64}[])
end

Tables.istable(::Type{DifferentiationBenchmark}) = true
Tables.rowaccess(::Type{DifferentiationBenchmark}) = true
Tables.columnaccess(::Type{DifferentiationBenchmark}) = false
Tables.rows(data::DifferentiationBenchmark) = data.rows

"""
    run_benchmark!(...)

Perform the actual measurement of preparation and differentiation efficiency.

!!! warning
    Implemented in a package extension that depends on [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl).
    If this function fails with a `MethodError`, try `import Chairmarks` before running it again.
"""
function run_benchmark! end
