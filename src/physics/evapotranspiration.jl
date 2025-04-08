function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu)
    # Compute a²[n] and c
    a_squared = (K^2) ./ (log.((z2 .- d0_gpu) ./ z0_gpu).^2)

    println("LOG TERM!: ", minimum(log.((z2 .- d0_gpu) ./ z0_gpu)), " / ", maximum(log.((z2 .- d0_gpu) ./ z0_gpu)))
    println("z2: ", minimum(z2), " / ", maximum(z2))
    println("K: ", minimum(K), " / ", maximum(K))
    println("d0_gpu: ", minimum(d0_gpu), " / ", maximum(d0_gpu))
    println("z0_gpu: ", minimum(z0_gpu), " / ", maximum(z0_gpu))

    c_coefficient = 49.82 .* a_squared .* sqrt.((z2 .- d0_gpu) ./ z0_gpu)
    
    # Compute Richardson number
    # NOTE TODO:
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

    println("a_squared: ", minimum(a_squared), " / ", maximum(a_squared))
    println("Fw: ", minimum(Fw), " / ", maximum(Fw))

    println("transfer_coefficient: ", minimum(transfer_coefficient), " / ", maximum(transfer_coefficient))
    println("wind_gpu: ", minimum(wind_gpu), " / ", maximum(wind_gpu))


    println("aerodynamic_resistance: ", minimum(aerodynamic_resistance), " / ", maximum(aerodynamic_resistance))

    return aerodynamic_resistance
end

function compute_partial_canopy_resistance(rmin_gpu, LAI_gpu)
    # Canopy resistance based on soil moisture (Eq. 6), without gsm multiplication; 
    # done in evapotranspiration calculation step   

    # Perform element-wise division with correct shapes
    return rmin_gpu ./ LAI_gpu
end

function calculate_net_radiation(swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
    return (1.0 .- albedo_gpu) .* swdown_gpu .+ emissivity .* (lwdown_gpu .- sigma .* (tsurf .+ 273.15).^4)
end

function calculate_potential_evaporation(tair_gpu, vp_gpu, elev_gpu, net_radiation, aerodynamic_resistance, rarc_gpu)
    # Compute intermediate variables
    vpd           = calculate_vpd(tair_gpu, vp_gpu) # [Pa]
    println("vpd[50,10] = ", Array(vpd)[50, 10])

    slope         = calculate_svp_slope(tair_gpu) # [Pa/°C]
    println("slope[50,10] = ", Array(slope)[50, 10])

    latent_heat   = calculate_latent_heat(tair_gpu) # [J/kg]
    println("latent_heat[50,10] = ", Array(latent_heat)[50, 10])

    scale_height  = calculate_scale_height(tair_gpu, elev_gpu) # [m] 
    println("scale_height[50,10] = ", Array(scale_height)[50, 10])

    surface_pressure = p_std .* exp.(-elev_gpu ./ scale_height) # [Pa]
    println("surface_pressure[50,10] = ", Array(surface_pressure)[50, 10])

    psychrometric_constant = 1628.6 .* surface_pressure ./ latent_heat # [Pa/K]
    println("psychrometric_constant[50,10] = ", Array(psychrometric_constant)[50, 10])

    air_density    = 0.003486 .* surface_pressure ./ (273.15 .+ tair_gpu) # [kg/m^3]
    println("air_density[50,10] = ", Array(air_density)[50, 10])


    # Penman-Monteith equation (mm/day) with canopy resistance set to zero
    numerator = slope .* (net_radiation .* day_sec) .+ (air_density .* c_p_air .* vpd ./ aerodynamic_resistance)
    denominator = latent_heat .* (slope .+ psychrometric_constant .* (1 .+ (rarc_gpu) ./ aerodynamic_resistance))
    potential_evaporation = (numerator ./ denominator)  # kg/m^2/day = [mm/day]

    # Ensure evaporation is non-negative when vapor pressure deficit is positive
    potential_evaporation = ifelse.((vpd .>= 0.0) .& (potential_evaporation .< 0.0),
                                    0.0,
                                    potential_evaporation)
    
    return potential_evaporation # [mm/day]
end



function calculate_max_water_storage(LAI_gpu)
    # Compute maximum water intercepted/stored in the canopy cover
    return K_L .* LAI_gpu  
end

function calculate_canopy_evaporation(water_storage, max_water_storage, potential_evaporation, aerodynamic_resistance, rarc, prec_gpu)

    # Compute potential canopy evaporation
    canopy_evaporation_star = (water_storage ./ max_water_storage).^(2 / 3) .* potential_evaporation .* 
                              (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc))

    # Ensure canopy evaporation fraction (f_n) is bounded between 0 and 1
    f_n = min.(1.0, (water_storage .+ prec_gpu) ./ canopy_evaporation_star)

    # Compute actual canopy evaporation
    canopy_evaporation = f_n .* canopy_evaporation_star

    return canopy_evaporation
end


function calculate_transpiration(
    potential_evaporation::CuArray, aerodynamic_resistance::CuArray, rarc_gpu::CuArray,  
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray, soil_moisture_critical::CuArray, 
    wilting_point::CuArray, root_gpu::CuArray
)
    println("soil_moisture_old shape: ", size(soil_moisture_old))
    println("soil_moisture_critical shape: ", size(soil_moisture_critical))
    println("wilting_point shape: ", size(wilting_point))
    println("root_gpu shape: ", size(root_gpu))

    # Compute stress factor
    #gsm_inv = calculate_gsm_inv(soil_moisture_old, soil_moisture_critical, wilting_point)
    #println("gsm_inv shape: ", size(gsm_inv))

    # Expand for broadcasting
   # gsm_inv_exp = reshape(gsm_inv, size(gsm_inv,1), size(gsm_inv,2), size(gsm_inv,3), 1)
    #println("g_sm_exp shape: ", size(g_sm_exp))

    canopy_resistance = compute_partial_canopy_resistance(rmin_gpu, LAI_gpu) # ./ gsm_inv # Eq. (6), TODO: add the gsm_inv multiplication

    ## Calculate modifier factor
    transpiration = (1.0 .- (water_storage ./ max_water_storage) .^ (2/3)) .* potential_evaporation .* (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc_gpu .+ canopy_resistance))
#
    ## Compute layer-wise transpiration
    #E_layer = factor .* (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc_gpu .+ canopy_resistance_exp ./ g_sm_exp))
    #println("E_layer shape: ", size(E_layer))
#
    ## Weighted sum with root fraction
    #E_t = sum(root_gpu .* E_layer, dims=3)
    #println("E_t shape (final transpiration): ", size(E_t))

    return transpiration
end


function calculate_soil_evaporation(soil_moisture, soil_moisture_max, potential_evaporation, b_i)
    # Compute the saturated area fraction
    A_sat = 1.0 .- (1.0 .- soil_moisture ./ soil_moisture_max).^b_i
    
    # Compute the unsaturated area fraction
    x = 1.0 .- A_sat

    # Approximate the series expansion from eq. (15) using the first four terms
    S_series = 1.0 .+
               (b_i ./ (1.0 .+ b_i)) .* x.^(1.0 ./ b_i) .+
               (b_i ./ (2.0 .+ b_i)) .* x.^(2.0 ./ b_i) .+
               (b_i ./ (3.0 .+ b_i)) .* x.^(3.0 ./ b_i)
    
    i_m = (1.0 .+ b_i).*soil_moisture_max # Max infiltration, Eq. 17 rewritten
    i_0 = i_m .* (1.0 .- (1.0 .- A_sat).^(1.0 ./ b_i)) # Eq. 13, TODO: check if correct, this is how it's done in VIC-WUR

    # This factor adjusts the unsaturated evaporation contribution
    Ev_unsat = potential_evaporation[:,:,:,end] .* i_0 ./ i_m .* x .* S_series
    
    # Evaporation from the saturated fraction occurs at the full potential rate
    Ev_sat = potential_evaporation[:,:,:,end] .* A_sat

    # Total soil evaporation is the sum of saturated and unsaturated contributions
    return Ev_sat .+ Ev_unsat
end
