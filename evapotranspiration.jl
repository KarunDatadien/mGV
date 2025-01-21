function apply_gpu_transformations!(precipitation_gpu::CuArray{Float32}, tair_gpu::CuArray{Float32}, iterations::Int)
    for _ in 1:iterations
        precipitation_gpu .= sqrt.(max.(precipitation_gpu, 0) .+ 1)       # Square root transformation
        precipitation_gpu .= log.(precipitation_gpu .+ 1e-6)              # Logarithmic transformation
        precipitation_gpu .= exp.(precipitation_gpu) .* sin.(precipitation_gpu)  # Non-linear exp and sine combination
        tair_gpu .= tair_gpu .+ exp.(precipitation_gpu) .* sin.(tair_gpu)
    end
end

function apply_cpu_transformations!(precipitation_cpu::Array{Float32}, tair_cpu::Array{Float32}, iterations::Int)
    for _ in 1:iterations
        # Square root transformation
        precipitation_cpu .= sqrt.(max.(precipitation_cpu, 0) .+ 1)
        
        # Logarithmic transformation
        precipitation_cpu .= log.(precipitation_cpu .+ 1e-6)
        
        # Non-linear exp and sine combination
        precipitation_cpu .= exp.(precipitation_cpu) .* sin.(precipitation_cpu)
        
        # Update tair_cpu using precipitation_cpu
        tair_cpu .= tair_cpu .+ exp.(precipitation_cpu) .* sin.(tair_cpu)
    end
end
