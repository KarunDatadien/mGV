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

# ----

function compute_c(z2, d0::CuArray{Float32}, z0::CuArray{Float32}, K)
    # Compute a²[n] using element-wise operations
    a_squared = (K^2) ./ (log.((z2 .- d0) ./ z0).^2)
    
    # Compute c using element-wise operations
    c = 49.82f0 .* a_squared .* sqrt.((z2 .- d0) ./ z0)
    
    return c
end

function compute_Fw(Ri_B, c)
    """
    Computes Fw[n] based on the given Ri_B[n] and constant c.

    Arguments:
        Ri_B :: Array of Richardson numbers (Ri_B[n]).
        c    :: Constant used in the equation.

    Returns:
        Fw   :: Array of Fw[n] values computed based on the conditions.
    """
    Fw = similar(Ri_B)  # Create an array to store results with the same shape as Ri_B

    # Compute Fw[n] for Ri_B[n] < 0
    Fw[Ri_B .< 0] .= 1 .- (9.4 .* Ri_B[Ri_B .< 0]) ./ 
                          (1 .+ c .* abs.(Ri_B[Ri_B .< 0]).^0.5)

    # Compute Fw[n] for 0 ≤ Ri_B[n] ≤ 0.2
    Fw[(Ri_B .>= 0) .& (Ri_B .<= 0.2)] .= 1 ./ 
        (1 .+ 4.7 .* Ri_B[(Ri_B .>= 0) .& (Ri_B .<= 0.2)]).^2

    return Fw
end
