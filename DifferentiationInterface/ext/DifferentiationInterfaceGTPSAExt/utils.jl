function initialize!(xt::TPS, x::Union{Number,Nothing}, dx::Union{Number,Nothing}, varidx::Integer=1)
    if !isnothing(x)
        xt[0] = x
    end
    if !isnothing(dx) 
        xt[varidx] = dx
    end
    return xt
end


function initialize!(xt::AbstractArray{TPS{T}}, x::AbstractArray, dx::AbstractArray, varidx=1) where {T}
    if varidx isa Number
        for i in eachindex(xt, x, dx, dx)
            initialize!(xt[i], x[i], dx[i], varidx)
        end
    else
        for i in eachindex(xt, x, dx, varidxs)
            initialize!(xt[i], x[i], dx[i], varidx[i])
        end
    end
end
#=
function initialize!(xt::AbstractArray{TPS{T}}, x::AbstractArray, dx::AbstractArray, varidx=1) where {T}
    if varidx isa Number
        for i in eachindex(xt, x, dx, dx)
            initialize!(xt[i], x[i], dx[i], varidx)
        end
    else
        for i in eachindex(xt, x, dx, varidxs)
            initialize!(xt[i], x[i], dx[i], varidx[i])
        end
    end
end
=#
