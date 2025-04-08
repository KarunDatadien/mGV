function calculate_surface_runoff(prec_gpu, soil_moisture_old, soil_moisture_max, b_i)
    # Sum the soil moisture and maximum soil moisture across the top two layers (layer 1)
    topsoil_moisture = sum(soil_moisture_old[:, :, 1:2], dims=3)
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:2], dims=3)

    # Compute the saturated area fraction (A_sat) as in the evaporation code
    A_sat = 1.0 .- (1.0 .- topsoil_moisture ./ topsoil_moisture_max) .^ b_i

    # Compute i_m (maximum infiltration capacity) using Eq. 17 rewritten
    i_m = (1.0 .+ b_i) .* topsoil_moisture_max

    # Compute i_0 (initial infiltration capacity) using Eq. 13
    i_0 = i_m .* (1.0 .- (1.0 .- A_sat) .^ (1.0 ./ b_i))

    # Compute the total water input: initial infiltration capacity + precipitation
    # Assuming prec_gpu is P * Δt, and Δt = 1 for simplicity
    total_water_input = i_0 .+ prec_gpu[:, :, :, end]

    # Compute runoff (Q_d * Δt) using ifelse
    # If total_water_input >= i_m, use Eq. 18a, otherwise use Eq. 18b
    runoff = ifelse.(total_water_input .>= i_m,
                     # Eq. 18a: Saturated case
                     prec_gpu[:, :, :, end] .- topsoil_moisture_max .+ topsoil_moisture,
                     # Eq. 18b: Unsaturated case
                     prec_gpu[:, :, :, end] .- topsoil_moisture_max .+ topsoil_moisture .+
                     topsoil_moisture_max .* (1.0 .- total_water_input ./ i_m) .^ (1.0 .+ b_i))

    # Ensure runoff is non-negative (numerical errors can cause small negative values)
    runoff = max.(runoff, 0.0)

    return runoff
end

function calculate_subsurface_runoff(
    soil_moisture_old, soil_moisture_max, Ds_gpu, Dsmax_gpu, Ws_gpu
)
    # Layer 2 is the third layer (index 3), representing the lower soil layer  
    bottomsoil_moisture = soil_moisture_old[:, :, 3:3]  # W_2^-[N+1], shape (204, 180, 1)
    bottomsoil_moisture_max = soil_moisture_max[:, :, 3:3]   # W_2^c, shape (204, 180, 1)
    Ws_fraction = Ws_gpu .* bottomsoil_moisture_max         # W_s * W_2^c, shape (204, 180, 1)

    # Initialize subsurface runoff (Q_b * Δt, assuming Δt = 1 day)
    Q_b = CUDA.zeros(Float32, 204, 180, 1)

    # Compute subsurface runoff using ifelse for Eq. 21a and 21b
    Q_b = ifelse.(
        bottomsoil_moisture .<= Ws_fraction,
        # Eq. 21a: Linear drainage
        (Ds_gpu .* Dsmax_gpu ./ Ws_fraction) .* bottomsoil_moisture,
        # Eq. 21b: Nonlinear drainage
        (Ds_gpu .* Dsmax_gpu ./ Ws_fraction) .* bottomsoil_moisture .+
        (Dsmax_gpu .- (Ds_gpu .* Dsmax_gpu ./ Ws_gpu)) .* 
        ((bottomsoil_moisture .- Ws_fraction) ./ (bottomsoil_moisture_max .- Ws_fraction)) .^ 2
    )

    # Ensure non-negative runoff
    Q_b = max.(Q_b, 0.0)

    return Q_b
end