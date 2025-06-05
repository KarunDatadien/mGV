function calculate_surface_runoff(prec_gpu, throughfall, soil_moisture_old, soil_moisture_max, b_i, cv_gpu)
    # Sum the soil moisture and maximum soil moisture across the top two layers (layer 1)
    topsoil_moisture = sum(soil_moisture_old[:, :, 1:2], dims=3)
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:2], dims=3)

    # Compute the saturated area fraction (A_sat) as in the evaporation code
    A_sat = 1.0 .- (1.0 .- topsoil_moisture ./ topsoil_moisture_max) .^ b_i

    # Compute i_m (maximum infiltration capacity) using Eq. 17 rewritten
    i_m = (1.0 .+ b_i) .* topsoil_moisture_max

    # Compute i_0 (initial infiltration capacity) using Eq. 13
    i_0 = i_m .* (1.0 .- (1.0 .- A_sat) .^ (1.0 ./ b_i))

    # Compute runoff for vegetated layers (n = 1 to nveg) using throughfall
    total_water_input_veg = i_0 .+ sum_with_nan_handling(throughfall[:, :, :, 1:end-1], 4)
    runoff_veg = ifelse.(total_water_input_veg  .>= i_m ,
                         # Eq. 18a: Saturated case
                         throughfall .- topsoil_moisture_max .+ topsoil_moisture,
                         # Eq. 18b: Unsaturated case
                         throughfall .- topsoil_moisture_max .+ topsoil_moisture .+
                         topsoil_moisture_max .* (1.0 .- total_water_input_veg ./ i_m) .^ (1.0 .+ b_i))

    runoff = runoff_veg
    
    # Ensure runoff is non-negative
    runoff = max.(runoff, 0.0)

    runoff = sum_with_nan_handling(runoff, 4)

    return runoff
end

function calculate_subsurface_runoff(soil_moisture_old, soil_moisture_max, Ds_gpu, Dsmax_gpu, Ws_gpu)
    bottomsoil_moisture = soil_moisture_old[:, :, 3:3]  # W_2^-[N+1], shape (204, 180, 1)
    bottomsoil_moisture_max = soil_moisture_max[:, :, 3:3]   # W_2^c, 
    Ws_fraction = Ws_gpu .* bottomsoil_moisture_max         # W_s * W_2^c, shape (204, 180, 1)

    # Initialize subsurface runoff (Q_b * Δt, assuming Δt = 1 day)
    Q_b = CUDA.zeros(float_type, size(bottomsoil_moisture, 1), size(bottomsoil_moisture, 2), size(bottomsoil_moisture, 3))

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

    Q_b = max.(Q_b, 0.0)     # Ensure non-negative runoff

    return Q_b
end

# Eq. (24): Total runoff
function calculate_total_runoff(surface_runoff, subsurface_runoff, cv_gpu)

    surface_runoff .= ifelse.(isnan.(surface_runoff) .| (abs.(surface_runoff) .> fillvalue_threshold), 0.0, surface_runoff) # Q_d[n]
    subsurface_runoff .= ifelse.(isnan.(subsurface_runoff) .| (abs.(subsurface_runoff) .> fillvalue_threshold), 0.0, subsurface_runoff) # Q_b[n]

    # Sum surface and subsurface runoff, weighted by coverage
 #   total_runoff = sum_with_nan_handling(cv_gpu .* (surface_runoff .+ subsurface_runoff), 4) # C_v[n]
     total_runoff = (surface_runoff .+ subsurface_runoff) # TODO: with or without cv_gpu?

    return total_runoff
end