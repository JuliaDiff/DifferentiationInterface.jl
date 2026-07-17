## Backend counterparts

DI.forward_counterpart(backend::AutoMooncake) = AutoMooncakeForward(; config = backend.config)
DI.reverse_counterpart(backend::AutoMooncakeForward) = AutoMooncake(; config = backend.config)
