stack_vec_col(t::NTuple) = stack(vec, t; dims=2)
stack_vec_row(t::NTuple) = stack(vec, t; dims=1)

ismutable_array(::Type) = true
ismutable_array(x) = ismutable_array(typeof(x))
