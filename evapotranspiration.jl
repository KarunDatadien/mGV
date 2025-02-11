function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu)
    # Compute a²[n] and c
    a_squared = (K^2) ./ (log.((z2 .- d0_gpu) ./ z0_gpu).^2)
    c_coefficient = 49.82f0 .* a_squared .* sqrt.((z2 .- d0_gpu) ./ z0_gpu)
    
    # Compute Richardson number
    # NOTE:
    # - Ri_B and Fw are currently allocated as Float64.
    # - It would be better to preallocate them at the beginning of the simulation as Float32
    #   (assuming all other arrays are also Float32).
    # - Then use `Fw .=` and `Ri_B .=` to overwrite their contents.
    Ri_B = ifelse.(
        tsurf .!= tair_gpu,
        g .* (tair_gpu .- tsurf) .* (z2 .- d0_gpu) ./ 
        (((tair_gpu .+ t_freeze) .+ (tsurf .+ t_freeze)) ./ 2 .* wind_gpu.^2),
        0.0
    )

    Ri_B = clamp.(Ri_B, -0.5, Ri_cr)
 
    # Compute friction factor
    Fw = ifelse.(Ri_B .< 0,
         1 .- (9.4 .* Ri_B) ./ (1 .+ c_coefficient .* abs.(Ri_B).^0.5),
         1 ./ (1 .+ 4.7 .* Ri_B).^2
    ) 

    # Compute transfer coefficient and aerodynamic resistance
    transfer_coefficient = 1.351 .* a_squared .* Fw
    aerodynamic_resistance = 1 ./ (transfer_coefficient .* wind_gpu)

    return aerodynamic_resistance
end

function compute_canopy_resistance(rmin_gpu, LAI_gpu)
    # Canopy resistance based on soil moisture (Eq. 6), without gsm multiplication; 
    # done in evapotranspiration calculation step   

    # Ensure rmin_gpu has dimensions (4320, 1680, 1, 14) for correct broadcasting
    rmin_reshaped = reshape(rmin_gpu, size(rmin_gpu, 1), size(rmin_gpu, 2), 1, size(rmin_gpu, 3))
    
    # Perform element-wise division with correct shapes
    return rmin_reshaped ./ LAI_gpu
end

function calculate_net_radiation(swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
    return (1 .- albedo_gpu) .* swdown_gpu .+ emissivity .* (lwdown_gpu .- sigma .* tsurf.^4)
end

function calculate_potential_evaporation(tair_gpu, vp_gpu, elev_gpu, net_radiation, aerodynamic_resistance, rc, rarc_gpu)

    # Reshape rarc_gpu to match (4320, 1680, 1, 14) before computation
    rarc_reshaped = reshape(rarc_gpu, size(rarc_gpu, 1), size(rarc_gpu, 2), 1, size(rarc_gpu, 3))

    vpd = calculate_vpd(tair_gpu,vp_gpu)
    slope = calculate_svp_slope(tair_gpu)
    latent_heat = calculate_latent_heat(tair_gpu)
    scale_height = calculate_scale_height(tair_gpu, elev_gpu)

    surface_pressure = p_std .* exp.(-elev_gpu ./ scale_height)
    
    psychrometric_constant = 1628.6 .* surface_pressure ./ latent_heat
    air_density = 0.003486 .* surface_pressure ./ (275 .+ tair_gpu)
    
    # Penman-Monteith equation
    potential_evaporation = (slope .* net_radiation .+ air_density .* c_p_air .* vpd ./ aerodynamic_resistance) ./ 
                            (latent_heat .* (slope .+ psychrometric_constant .* (1 .+ (rc .+ rarc_reshaped) ./ aerodynamic_resistance))) .* day_sec
    
    # Ensure evaporation is non-negative when vapor pressure deficit is positive
    potential_evaporation = ifelse.((vpd .>= 0.0) .& (potential_evaporation .< 0.0), 0.0, potential_evaporation)
    
    return potential_evaporation
end


function calculate_max_water_storage(LAI_gpu)
    # Compute maximum water intercepted/stored in the canopy cover
    return K_L .* LAI_gpu  
end

function calculate_canopy_evaporation(water_storage, max_water_storage, LAI_gpu, potential_evaporation, aerodynamic_resistance, rarc, prec_gpu)

    # Compute potential canopy evaporation
    canopy_evaporation_star = (water_storage ./ max_water_storage).^(2 / 3) .* potential_evaporation .* 
                              (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc))

    # Ensure canopy evaporation fraction (f_n) is bounded between 0 and 1
    f_n = min.(1.0, (water_storage .+ prec_gpu) ./ canopy_evaporation_star)

    # Compute actual canopy evaporation
    canopy_evaporation = f_n .* canopy_evaporation_star

    return canopy_evaporation
end