# Dummy functions:
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

function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu)
    # Compute a²[n] and c
    a_squared = (K^2) ./ (log.((z2 .- d0_gpu) ./ z0_gpu).^2)
    c_coefficient = 49.82f0 .* a_squared .* sqrt.((z2 .- d0_gpu) ./ z0_gpu)
    
    # Compute Richardson number
    Ri_B = ifelse.(
        tsurf .!= tair_gpu,
        g .* (tair_gpu .- tsurf) .* (z2 .- d0_gpu) ./ 
        (((tair_gpu .+ t_freeze) .+ (tsurf .+ t_freeze)) ./ 2 .* wind_gpu.^2),
        0.0
    )

    Ri_B = clamp!.(Ri_B, -0.5, Ri_cr)
    
    # Compute friction factor
    Fw = similar(Ri_B) 
    Fw[Ri_B .< 0] .= 1 .- (9.4 .* Ri_B[Ri_B .< 0]) ./ 
                          (1 .+ c_coefficient .* abs.(Ri_B[Ri_B .< 0]).^0.5)
    Fw[(Ri_B .>= 0) .& (Ri_B .<= Ri_cr)] .= 1 ./ 
        (1 .+ 4.7 .* Ri_B[(Ri_B .>= 0) .& (Ri_B .<= Ri_cr)]).^2
    
    # Compute transfer coefficient and aerodynamic resistance
    transfer_coefficient = 1.351 * a_squared * Fw
    aerodynamic_resistance = 1 ./ (transfer_coefficient .* wind_gpu)
    
    return aerodynamic_resistance
end