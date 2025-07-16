function calculate_surface_runoff(prec_gpu, throughfall, soil_moisture_old, soil_moisture_max, b_i, cv_gpu)
    # Sum top two layers (upper zone in VIC/Liang 1994)
    topsoil_moisture = sum(soil_moisture_old[:, :, 1:2], dims=3)[:,:,1]  # Make 2D
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:2], dims=3)[:,:,1]

    # Saturated area fraction (Eq. preceding 17 in Liang 1994)
    A_sat = 1.0 .- (1.0 .- topsoil_moisture ./ topsoil_moisture_max) .^ b_i

    # Max infiltration capacity (rewritten Eq. 17)
    i_m = (1.0 .+ b_i) .* topsoil_moisture_max

    # Initial infiltration capacity (Eq. 13)
    i_0 = i_m .* (1.0 .- (1.0 .- A_sat) .^ (1.0 ./ b_i))

    # Total water input to soil (aggregated throughfall across all tiles)
    total_water_input = sum_with_nan_handling(throughfall, 4)  # mm over grid

    # Surface runoff (Eq. 18a/b in Liang 1994)
    total_water_input_veg = total_water_input .+ i_0  # Note: no separate veg here, as shared soil
    runoff = ifelse.(total_water_input_veg .>= i_m,
                     total_water_input .- (topsoil_moisture_max .- topsoil_moisture),
                     total_water_input .- topsoil_moisture_max .+ topsoil_moisture_max .* (1.0 .- total_water_input_veg ./ i_m) .^ (1.0 .+ b_i))

    # Ensure non-negative
    runoff = max.(runoff, 0.0)

    return runoff  # Single 2D array for grid-total surface runoff
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