function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu)    
    roughness = CUDA.zeros(float_type, size(cv_gpu))

    roughness[:, :, :, 1:end-1] = z0_gpu[:, :, :, 1:end-1] .* cv_gpu[:, :, :, 1:end-1]
    roughness[:, :, :, end:end] = z0soil_gpu .* cv_gpu[:, :, :, end:end]

    # Compute a²[n] and c
    a_squared = (K^2) ./ (log.((z2 .- d0_gpu) ./ roughness).^2)
    c_coefficient = 49.82 .* a_squared .* sqrt.((z2 .- d0_gpu) ./ roughness)
    
    # Compute Richardson number
    # NOTE TODO:
    # - Ri_B and Fw are currently allocated as Float64.
    # - It would be better to preallocate them at the beginning of the simulation as float_type
    #   (assuming all other arrays are also float_type).
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

function compute_partial_canopy_resistance(rmin_gpu, LAI_gpu)
    # Canopy resistance based on soil moisture (Eq. 6), without gsm multiplication; done in evapotranspiration calculation step   
    return rmin_gpu ./ LAI_gpu
end

function calculate_net_radiation(swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
    return (1.0 .- albedo_gpu) .* swdown_gpu .+ emissivity .* (lwdown_gpu .- sigma .* (tsurf .+ 273.15).^4)
end

function calculate_potential_evaporation(tair_gpu, vp_gpu, elev_gpu, net_radiation, aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu)
    # Compute intermediate variables
    vpd           = max.(calculate_vpd(tair_gpu, vp_gpu), 0.0)  # [Pa], ensure non-negative
    slope         = calculate_svp_slope(tair_gpu) # [Pa/°C]
    latent_heat   = calculate_latent_heat(tair_gpu) # [J/kg]
    scale_height  = calculate_scale_height(tair_gpu, elev_gpu) # [m] 
    surface_pressure = p_std .* exp.(-elev_gpu ./ scale_height) # [Pa]
    psychrometric_constant = 1628.6 .* surface_pressure ./ latent_heat # [Pa/K]
    air_density = 0.003486 .* surface_pressure ./ (273.15 .+ tair_gpu) # [kg/m^3]

    rc = rmin_gpu ./ LAI_gpu # TODO: should be done more exactly with a gsm_inv value

    # Penman-Monteith equation (mm/day) with canopy resistance set to rc
    numerator = slope .* (net_radiation .* day_sec) .+ (air_density .* c_p_air .* vpd .* day_sec ./ aerodynamic_resistance)
    denominator = latent_heat .* (slope .+ psychrometric_constant .* (1 .+ (rc .+ rarc_gpu) ./ aerodynamic_resistance))
    potential_evaporation = (numerator ./ denominator)  # kg/m^2/day = [mm/day]

    # Add bare soil potential evaporation
    SOIL_RARC = 100. # TODO: is this ok? taken from VIC
    aerodynamic_resistance_soil = aerodynamic_resistance # TODO: replace this value
    numerator_soil = slope .* (net_radiation .* day_sec) .+ (air_density .* c_p_air .* vpd .* day_sec ./ aerodynamic_resistance_soil)
    denominator_soil = latent_heat .* (slope .+ psychrometric_constant .* (1 .+ SOIL_RARC ./ aerodynamic_resistance_soil)) # rc = 0 for bare soil
    potential_evaporation_soil = (numerator_soil ./ denominator_soil)
    potential_evaporation[:, :, :, end:end] = potential_evaporation_soil[:, :, :, end:end]

    # Ensure potential evaporation is non-negative
    potential_evaporation = max.(potential_evaporation, 0.0)

    return potential_evaporation # [mm/day]
end

function calculate_max_water_storage(LAI_gpu, cv_gpu)
    # Compute maximum water intercepted/stored in the canopy cover
    result = K_L .* LAI_gpu .* cv_gpu #TODO should we multiply by .* cv_gpu ?
    return ifelse.(isnan.(result) .| (abs.(result) .> fillvalue_threshold), 0.0, result)
end

function calculate_canopy_evaporation(water_storage, max_water_storage, potential_evaporation, aerodynamic_resistance, rarc, prec_gpu, cv_gpu)

    potential_evaporation .= ifelse.(isnan.(potential_evaporation) .| (abs.(potential_evaporation) .> fillvalue_threshold), 0.0, potential_evaporation)
    water_storage .= ifelse.(isnan.(water_storage) .| (abs.(water_storage) .> fillvalue_threshold), 0.0, water_storage)
    max_water_storage .= ifelse.(isnan.(max_water_storage) .| (abs.(max_water_storage) .> fillvalue_threshold), 0.0, max_water_storage)
    aerodynamic_resistance .= ifelse.(isnan.(aerodynamic_resistance) .| (abs.(aerodynamic_resistance) .> fillvalue_threshold), 0.0, aerodynamic_resistance)
    rarc .= ifelse.(isnan.(rarc) .| (abs.(rarc) .> fillvalue_threshold), 0.0, rarc)

    # Compute potential canopy evaporation
    canopy_evaporation_star = (water_storage ./ max_water_storage).^(2 / 3) .* potential_evaporation .* 
                              (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc))

    # Ensure canopy evaporation fraction (f_n) is bounded between 0 and 1
    f_n = min.(1.0, (water_storage .+ prec_gpu .* cv_gpu) ./ canopy_evaporation_star)

    # Compute actual canopy evaporation
    canopy_evaporation = f_n .* canopy_evaporation_star
    canopy_evaporation = ifelse.(isnan.(canopy_evaporation) .| (abs.(canopy_evaporation) .> fillvalue_threshold), 0.0, canopy_evaporation)

    return canopy_evaporation
end

function calculate_transpiration(
    potential_evaporation::CuArray, aerodynamic_resistance::CuArray, rarc_gpu::CuArray,  
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray, 
    soil_moisture_critical::CuArray, wilting_point::CuArray, root_gpu::CuArray, 
    rmin_gpu::CuArray, LAI_gpu::CuArray, cv_gpu
)
    # Constants
    delta_t = 1  # seconds per day

    # Replace NaN or large values with 0.0
    potential_evaporation .= ifelse.(isnan.(potential_evaporation) .| (abs.(potential_evaporation) .> fillvalue_threshold), 0.0, potential_evaporation)
    water_storage .= ifelse.(isnan.(water_storage) .| (abs.(water_storage) .> fillvalue_threshold), 0.0, water_storage)
    max_water_storage .= ifelse.(isnan.(max_water_storage) .| (abs.(max_water_storage) .> fillvalue_threshold), 0.0, max_water_storage)
    aerodynamic_resistance .= ifelse.(isnan.(aerodynamic_resistance) .| (abs.(aerodynamic_resistance) .> fillvalue_threshold), 0.0, aerodynamic_resistance)
    rarc_gpu .= ifelse.(isnan.(rarc_gpu) .| (abs.(rarc_gpu) .> fillvalue_threshold), 0.0, rarc_gpu)

    # Compute soil moisture for layers 1 and 2
    W_1 = sum(soil_moisture_old[:, :, 1:2], dims=3)
    W_2 = soil_moisture_old[:, :, 3:3]

    # Critical and wilting points for layers
    W_1_cr = sum(soil_moisture_critical[:, :, 1:2], dims=3)
    W_2_cr = soil_moisture_critical[:, :, 3:3]
    W_1_star = sum(wilting_point[:, :, 1:2], dims=3)
    W_2_star = wilting_point[:, :, 3:3]

    # Root fractions
    f_1 = sum(root_gpu[:, :, 1:2, :], dims=3)
    f_2 = root_gpu[:, :, 3:3, :]

    # Normalize root fractions
    f_sum = f_1 .+ f_2 
    f_1_normalized = f_1 ./ f_sum
    f_2_normalized = f_2 ./ f_sum

    # Soil moisture stress factors
    g_sw_1 = ifelse.(W_1 .>= W_1_cr, 1.0,
                     ifelse.(W_1 .< W_1_star, 0.0,
                             (W_1 .- W_1_star) ./ (W_1_cr .- W_1_star)))
    g_sw_2 = ifelse.(W_2 .>= W_2_cr, 1.0,
                     ifelse.(W_2 .< W_2_star, 0.0,
                             (W_2 .- W_2_star) ./ (W_2_cr .- W_2_star)))

    # Weighted average stress factor
    g_sw = (f_1_normalized .* g_sw_1 .+ f_2_normalized .* g_sw_2) ./ (f_1_normalized .+ f_2_normalized)
    g_sw = clamp.(ifelse.(isnan.(g_sw), 0.0, g_sw), 0.0, 1.0)

    # A large resistance value to represent a closed canopy
    HUGE_R = 1e6

    # compute “both‐layers‐unstressed” mask
    both_ok = (W_1 .>= W_1_cr) .& (W_2 .>= W_2_cr)
    
    # use g_sw_case=1 where both layers are above crit; else keep your normal g_sw
    g_sw_case = ifelse.(both_ok, 1.0, g_sw)
    
    # now canopy resistance in pure ifelse style:
    canopy_resistance_1 = ifelse.(
        (LAI_gpu .<= 0.0) .| (g_sw_case .<= 0.0),  # no LAI or still totally dry
        HUGE_R,                                   # fully shut
        rmin_gpu ./ (LAI_gpu .* g_sw_case)        # Jarvis rmin/(LAI⋅g_sw_case)
    )

    # Canopy resistance
#    canopy_resistance_1 = ifelse.(LAI_gpu .== 0.0, 0.0, (rmin_gpu .* g_sw) ./ LAI_gpu)

    # Dry fraction
    dryFrac = ifelse.(max_water_storage .== 0.0, 0.0,
                      1.0 .- (water_storage ./ max_water_storage) .^ (2/3))

    # Total transpiration (mm/s to mm/day)
    transpiration = ifelse.(max_water_storage .== 0.0, 0.0, 
                           dryFrac .* potential_evaporation .* 
                           (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc_gpu .+ canopy_resistance_1)))
    transpiration = max.(transpiration, 0.0) .* delta_t

    # Distribute transpiration
    E_1_t = transpiration .* f_1_normalized
    E_2_t = transpiration .* f_2_normalized

    # Scaling factors
    scaling_1 = ifelse.(W_1 .>= W_1_cr, 1.0, max.(0.0, (W_1 .- W_1_star) ./ (W_1_cr .- W_1_star)))
    scaling_2 = ifelse.(W_2 .>= W_2_cr, 1.0, max.(0.0, (W_2 .- W_2_star) ./ (W_2_cr .- W_2_star)))

    # Temporary transpiration
    E_1_t_temp = E_1_t .* scaling_1
    E_2_t_temp = E_2_t .* scaling_2

    # Spare transpiration
    spare_transp = E_1_t .* (1.0 .- scaling_1) + E_2_t .* (1.0 .- scaling_2)

    # Updated root sum
    root_sum_updated = f_1_normalized .* (W_1 .>= W_1_cr) + f_2_normalized .* (W_2 .>= W_2_cr)

    # Reallocation factor
    realloc_factor = ifelse.(root_sum_updated .> 0.0, spare_transp ./ root_sum_updated, 0.0)

    # Final transpiration per layer
    E_1_t = E_1_t_temp + (W_1 .>= W_1_cr) .* f_1_normalized .* realloc_factor
    E_2_t = E_2_t_temp + (W_2 .>= W_2_cr) .* f_2_normalized .* realloc_factor

    # Final total transpiration
    transpiration = E_1_t + E_2_t
    transpiration = max.(transpiration, 0.0)
    E_1_t = max.(E_1_t, 0.0)
    E_2_t = max.(E_2_t, 0.0)

    return transpiration, E_1_t, E_2_t, g_sw_1, g_sw_2, g_sw
end


function calculate_soil_evaporation(soil_moisture, soil_moisture_max, potential_evaporation, b_i, cv_gpu)
    # Sum the soil moisture and maximum soil moisture across the top two layers
    topsoil_moisture = sum(soil_moisture[:, :, 1:1, end:end], dims=3)
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:1, end:end], dims=3)

    # Compute the saturated area fraction
    A_sat = 1.0 .- (1.0 .- topsoil_moisture ./ topsoil_moisture_max) .^ b_i

    # Compute the unsaturated area fraction
    x = 1.0 .- A_sat

    S_series = CUDA.ones(float_type, size(x))

    # Approximate the series expansion from eq. (15) using the first 30 terms
    for n = 1:30
        S_series .+= (b_i ./ (n .+ b_i)) .* x .^ (n ./ b_i)
    end

    i_m = (1.0 .+ b_i) .* topsoil_moisture_max  # Max infiltration, Eq. 17 rewritten
    i_0 = i_m .* (1.0 .- (1.0 .- A_sat) .^ (1.0 ./ b_i))  # Eq. 13

    # Unsaturated evaporation contribution
    Ev_unsat = potential_evaporation[:, :, :, end:end]  .* i_0 ./ i_m .* x .* S_series

    # Saturated evaporation contribution
    Ev_sat = potential_evaporation[:, :, :, end:end] .* A_sat # .* cv_gpu[:, :, :, end:end]

    # Total soil evaporation
    return (Ev_sat .+ Ev_unsat) 
end


function update_water_canopy_storage(water_storage, prec_gpu, cv_gpu, canopy_evaporation, max_water_storage, throughfall)

    # Calculate new water storage: current storage + (precipitation - canopy evaporation)
    new_water_storage = water_storage .+ (prec_gpu .* cv_gpu) .- canopy_evaporation
    
    # Compute throughfall: excess water beyond max storage
    throughfall = max.(0, new_water_storage .- max_water_storage)
    
    # Update water storage: clamp between 0 and max_water_storage
    water_storage = max.(0.0, min.(new_water_storage, max_water_storage))
    
    return (water_storage), throughfall # TODO: why does (water_storage ./ 2) give near perfect values?
end

# Eq. (23): Total evapotranspiration
function calculate_total_evapotranspiration(canopy_evaporation, transpiration, soil_evaporation, cv_gpu)
    # Sum canopy evaporation and transpiration for vegetated classes (n = 1:nveg-1)
    vegetated_et = cv_gpu[:, :, :, 1:end-1] .* (canopy_evaporation[:, :, :, 1:end-1] .+ transpiration[:, :, :, 1:end-1])
    
    # Add bare soil evaporation (n = nveg)
    bare_soil_et = cv_gpu[:, :, :, end:end] .* soil_evaporation
    
    # Total evapotranspiration (sum across cover classes)
    total_et = sum_with_nan_handling(vegetated_et, 4) .+ bare_soil_et
  
    return total_et
end