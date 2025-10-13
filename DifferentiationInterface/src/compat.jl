macro public(ex)
    return if VERSION >= v"1.11.0-DEV.469"
        args = if ex isa Symbol
            (ex,)
        elseif Base.isexpr(ex, :tuple)
            ex.args
        else
            error("something informative")
        end
        esc(Expr(:public, args...))
    else
        nothing
    end
end
