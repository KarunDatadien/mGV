function apply_gpu_transformations!(precipitation_gpu::CuArray{Float32}, tair_gpu::CuArray{Float32}, iterations::Int)
    for _ in 1:iterations
        precipitation_gpu .= sqrt.(max.(precipitation_gpu, 0) .+ 1)       # Square root transformation
        precipitation_gpu .= log.(precipitation_gpu .+ 1e-6)              # Logarithmic transformation
        precipitation_gpu .= exp.(precipitation_gpu) .* sin.(precipitation_gpu)  # Non-linear exp and sine combination
        tair_gpu .= tair_gpu .+ exp.(precipitation_gpu) .* sin.(tair_gpu)
    end
end
