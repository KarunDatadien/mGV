function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu)    
    roughness = CUDA.zeros(float_type, size(cv_gpu))

    roughness[:, :, :, 1:end-1] = z0_gpu[:, :, :, 1:end-1]
    roughness[:, :, :, end:end] = z0soil_gpu

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

    # For vegetation tiles (1:end-1): Unstressed potential transpiration (PM with r_c = rmin / LAI)
    rc_veg = rmin_gpu[:, :, :, 1:end-1] ./ LAI_gpu[:, :, :, 1:end-1]
    numerator_veg = slope .* (net_radiation[:, :, :, 1:end-1] .* day_sec) .+ (air_density .* c_p_air .* vpd .* day_sec ./ 50.0)
    denominator_veg = latent_heat .* (slope .+ psychrometric_constant .* (1 .+ (rc_veg .+ rarc_gpu[:, :, :, 1:end-1]) ./ 50.0))
    potential_evaporation_veg = (numerator_veg ./ denominator_veg)  # kg/m²/day = [mm/day]

    # For bare soil: Potential soil evaporation (PM with r_c = 0)
    SOIL_RARC = 100.0  # TODO: is this ok? taken from VIC
    aerodynamic_resistance_soil = aerodynamic_resistance[:, :, :, end:end]  # Use bare soil slice
    net_radiation_soil = net_radiation[:, :, :, end:end]  # Use bare soil slice for consistency
    numerator_soil = slope .* (net_radiation_soil .* day_sec) .+ (air_density .* c_p_air .* vpd .* day_sec ./ aerodynamic_resistance_soil)
    denominator_soil = latent_heat .* (slope .+ psychrometric_constant .* (1 .+ SOIL_RARC ./ aerodynamic_resistance_soil))
    potential_evaporation_soil = (numerator_soil ./ denominator_soil)

    # Combine
    potential_evaporation = CUDA.zeros(float_type, size(net_radiation))
    potential_evaporation[:, :, :, 1:end-1] = potential_evaporation_veg
    potential_evaporation[:, :, :, end:end] = potential_evaporation_soil

    # Ensure potential evaporation is non-negative
    potential_evaporation = max.(potential_evaporation, 0.0)

    return potential_evaporation # [mm/day]
end

function calculate_max_water_storage(LAI_gpu, cv_gpu, coverage_gpu)
    # Compute maximum water intercepted/stored in the canopy cover
    result = K_L .* LAI_gpu .* cv_gpu .* coverage_gpu #TODO should we multiply by .* cv_gpu ?
    return ifelse.(isnan.(result) .| (abs.(result) .> fillvalue_threshold), 0.0, result)
end

function calculate_canopy_evaporation(water_storage, max_water_storage, potential_evaporation, aerodynamic_resistance, rarc, prec_gpu, cv_gpu, rmin, LAI_gpu)

    potential_evaporation .= ifelse.(isnan.(potential_evaporation) .| (abs.(potential_evaporation) .> fillvalue_threshold), 0.0, potential_evaporation)
    water_storage .= ifelse.(isnan.(water_storage) .| (abs.(water_storage) .> fillvalue_threshold), 0.0, water_storage)
    max_water_storage .= ifelse.(isnan.(max_water_storage) .| (abs.(max_water_storage) .> fillvalue_threshold), 0.0, max_water_storage)
    aerodynamic_resistance .= ifelse.(isnan.(aerodynamic_resistance) .| (abs.(aerodynamic_resistance) .> fillvalue_threshold), 0.0, aerodynamic_resistance)
    rarc .= ifelse.(isnan.(rarc) .| (abs.(rarc) .> fillvalue_threshold), 0.0, rarc)

    # Approximate wet E_p from unstressed potential
    rc = rmin ./ LAI_gpu
    e_p_wet = potential_evaporation .* (50.0 .+ rarc .+ rc) ./ (50.0 .+ rarc)

    # Compute potential canopy evaporation
#    canopy_evaporation_star = (water_storage ./ max_water_storage).^(2 / 3) .* e_p_wet .* 
#                              (50.0 ./ (50.0 .+ rarc))

    canopy_evaporation_star = (water_storage ./ max_water_storage).^(2 / 3) .* potential_evaporation .* 
                              (50.0 ./ (50.0 .+ rarc))
    # Ensure canopy evaporation fraction (f_n) is bounded between 0 and 1
    f_n = min.(1.0, (water_storage .+ prec_gpu .* cv_gpu) ./ canopy_evaporation_star)

    # Compute actual canopy evaporation
    canopy_evaporation = f_n .* canopy_evaporation_star
    canopy_evaporation = ifelse.(isnan.(canopy_evaporation) .| (abs.(canopy_evaporation) .> fillvalue_threshold), 0.0, canopy_evaporation)

    return canopy_evaporation
end

#=
This function has been updated to fix the GPU type-compatibility error.

The problem was caused by mixing input arrays of different precisions
(Float64 and Float32). The fix is to explicitly convert the Float64
inputs to Float32 at the beginning of the function. This ensures all
subsequent calculations are type-stable and GPU-compatible.
=#

using CUDA
using Printf

function calculate_transpiration(
    potential_evaporation::CuArray, aerodynamic_resistance::CuArray, rarc_gpu::CuArray,
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray,
    soil_moisture_critical::CuArray, wilting_point::CuArray, root_gpu::CuArray,
    rmin_gpu::CuArray, LAI_gpu::CuArray, cv_gpu
)
    EPSILON = 1.0f-9

    # --- FIX: Enforce Float32 for type stability ---
    # Convert all floating point inputs to Float32 at the start.
    pot_evap_f32 = Float32.(potential_evaporation)
    aero_res_f32 = Float32.(aerodynamic_resistance)
    max_ws_f32 = Float32.(max_water_storage)
    W_cr_f32 = Float32.(soil_moisture_critical)
    W_wp_f32 = Float32.(wilting_point)
    ws_f32 = Float32.(water_storage)
    sm_old_f32 = Float32.(soil_moisture_old)


    # --- 1. Define Soil and Root Properties for Each Layer ---
    W_1 = view(sm_old_f32, :, :, 1)
    W_2 = view(sm_old_f32, :, :, 2)
    
    W_cr_1 = view(W_cr_f32, :, :, 1)
    W_cr_2 = view(W_cr_f32, :, :, 2)

    W_wp_1 = view(W_wp_f32, :, :, 1)
    W_wp_2 = view(W_wp_f32, :, :, 2)

    f_1 = sum(view(root_gpu, :, :, 1, :), dims=3)[:,:,1]
    f_2 = sum(view(root_gpu, :, :, 2, :), dims=3)[:,:,1]

    # --- 2. Calculate Individual Layer Stress Factors (g_sm) ---
    g_sm_1 = (W_1 .- W_wp_1) ./ (W_cr_1 .- W_wp_1 .+ EPSILON)
    g_sm_2 = (W_2 .- W_wp_2) ./ (W_cr_2 .- W_wp_2 .+ EPSILON)
    
    g_sm_1 = clamp.(g_sm_1, 0.0f0, 1.0f0)
    g_sm_2 = clamp.(g_sm_2, 0.0f0, 1.0f0)

    # --- 3. Determine the Final Soil Moisture Stress (g_sw) ---
    g_sw = (f_1 .* g_sm_1 .+ f_2 .* g_sm_2) ./ (f_1 .+ f_2 .+ EPSILON)

    g_sw = ifelse.((W_2 .>= W_cr_2) .& (f_2 .>= 0.5f0), 1.0f0, g_sw)

    g_sw = ifelse.((W_1 .>= W_cr_1) .& (f_1 .>= 0.5f0), 1.0f0, g_sw)

    g_sw = clamp.(ifelse.(isnan.(g_sw), 0.0f0, g_sw), 0.0f0, 1.0f0)

    # --- 4. Calculate Total Transpiration ---
    dryFrac = 1.0f0 .- (ws_f32 ./ (max_ws_f32 .+ EPSILON)) .^ (2.0f0/3.0f0)
    dryFrac = clamp.(dryFrac, 0.0f0, 1.0f0)
    
    total_transpiration = dryFrac .* pot_evap_f32 .* g_sw
    total_transpiration = clamp.(total_transpiration, 0.0f0, Inf32)

    # --- 5. Distribute Transpiration to Each Layer ---
    denom = (f_1 .* g_sm_1 .+ f_2 .* g_sm_2 .+ EPSILON)
    E_1_t = total_transpiration .* (f_1 .* g_sm_1) ./ denom
    E_2_t = total_transpiration .* (f_2 .* g_sm_2) ./ denom

    E_1_t = clamp.(ifelse.(isnan.(E_1_t), 0.0f0, E_1_t), 0.0f0, total_transpiration)
    E_2_t = clamp.(ifelse.(isnan.(E_2_t), 0.0f0, E_2_t), 0.0f0, total_transpiration)

    return total_transpiration, E_1_t, E_2_t, g_sm_1, g_sm_2, g_sw
end




#TODO: investigate issue and cleanup function below
# Issue: SOIL EVAP WARNING: Found 402 cells where topsoil_moisture_max is zero. This will cause division by zero.

# You might need to import the Printf module to use @printf
using Printf

function calculate_soil_evaporation(soil_moisture, soil_moisture_max, potential_evaporation, b_i, cv_gpu)
    # A small epsilon to prevent division by zero in a numerically stable way.
    # Using a value appropriate for Float32.
    EPSILON = 1.0f-9 

    # Sum the soil moisture and maximum soil moisture across the top layer(s)
    topsoil_moisture = sum(soil_moisture[:, :, 1:1, end:end], dims=3)
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:1, end:end], dims=3)

    # ====================================================================
    # DIAGNOSTIC CHECKS: Find and report the root causes of instability
    # ====================================================================
    
    # Check 1: Detect potential for division by zero.
    # We count how many cells have a max capacity of zero.
    num_zero_max = CUDA.sum(topsoil_moisture_max .== 0.0f0)
    if num_zero_max > 0
        # Print a warning if this condition is met.
        @printf("SOIL EVAP WARNING: Found %d cells where topsoil_moisture_max is zero. This will cause division by zero.\n", num_zero_max)
    end

    # Check 2: Detect moisture ratios > 1. This is the most common cause of NaNs
    # in power functions like the one used for A_sat. It happens when
    # `topsoil_moisture` slightly exceeds `topsoil_moisture_max` due to floating point errors.
    num_ratio_gt_one = CUDA.sum(topsoil_moisture .> topsoil_moisture_max)
    if num_ratio_gt_one > 0
        @printf("SOIL EVAP WARNING: Found %d cells where moisture > max_moisture. This leads to taking the power of a negative number.\n", num_ratio_gt_one)
    end

    # ====================================================================
    # NUMERICAL SAFEGUARDS: Apply fixes to prevent NaNs
    # ====================================================================

    # Safeguard 1: Add epsilon to the denominator and clamp the ratio.
    # This simultaneously prevents division-by-zero and ensures the base of the
    # power function below is never negative.
    moisture_ratio = topsoil_moisture ./ (topsoil_moisture_max .+ EPSILON)
    safe_moisture_ratio = clamp.(moisture_ratio, 0.0f0, 1.0f0)

    # Compute the saturated area fraction using the sanitized ratio.
    A_sat = 1.0f0 .- (1.0f0 .- safe_moisture_ratio) .^ b_i
    # Clamp the result as an extra precaution.
    A_sat = clamp.(A_sat, 0.0f0, 1.0f0)

    # Compute the unsaturated area fraction
    x = 1.0f0 .- A_sat

    # Initialize S_series with ones.
    S_series = CUDA.ones(float_type, size(x))

    # Approximate the series expansion. This is now safe because `x` is guaranteed non-negative.
    for n = 1:30
        S_series .+= (b_i ./ (n .+ b_i)) .* x .^ (n ./ b_i)
    end

    # Safeguard 2: Calculate the infiltration ratio directly and safely.
    # Since A_sat is clamped, this calculation is now safe from NaNs.
    infiltration_ratio = 1.0f0 .- (1.0f0 .- A_sat) .^ (1.0f0 ./ b_i)

    # Unsaturated evaporation contribution
    Ev_unsat = potential_evaporation[:, :, :, end:end] .* infiltration_ratio .* x .* S_series

    # Saturated evaporation contribution
    Ev_sat = potential_evaporation[:, :, :, end:end] .* A_sat

    # Total soil evaporation
    return Ev_sat .+ Ev_unsat
end



function update_water_canopy_storage(water_storage, prec_gpu, cv_gpu, canopy_evaporation, max_water_storage, throughfall, coverage)

    # Calculate new water storage: current storage + (precipitation - canopy evaporation)
#    new_water_storage = water_storage .+ (prec_gpu .* cv_gpu .* coverage) .- canopy_evaporation
    new_water_storage = water_storage .+ (prec_gpu .* cv_gpu .* coverage) .- canopy_evaporation

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