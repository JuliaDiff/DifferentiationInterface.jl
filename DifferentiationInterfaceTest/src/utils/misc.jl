mysimilar(x::AbstractArray) = similar(x)

mycopy_random(x) = mycopy_random(default_rng(), x)
mycopy_random(rng::AbstractRNG, x::Number) = randn(rng, typeof(x))
mycopy_random(rng::AbstractRNG, x::AbstractArray) = map(Base.Fix1(mycopy_random, rng), x)
