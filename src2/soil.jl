include("runoff.jl")

"""
Scalar Physics Kernel (Inner Function)
"""
function soil_evap_kernel(sm_top, sm_max_top, resid_top, pe, b_i, cv, cov)
    # 1. Calculate Max Infiltration
    max_infil = (1f0 + b_i) * sm_max_top
    
    # 2. Moisture Ratio
    # This fraction determines how full the theoretical soil column is.
    ratio = clamp(1f0 - sm_top / sm_max_top, 0f0, 1f0)
    
    # 3. Handle b_i == -1.0 case without explicit branching
    # In VIC, if the shape parameter b_i is exactly -1.0, it denotes a linear storage limit.
    ratio_adj = b_i == -1f0 ? ratio : ratio ^ (1f0 / (b_i + 1f0))
    tmp = b_i == -1f0 ? max_infil : max_infil * (1f0 - ratio_adj)

    # 4. Saturation Check
    is_saturated = tmp >= max_infil
    
    # 5. ARNO Evaporation Logic
    # We replace the recursive if/else thresholds with branchless ternary expressions to compute the beta scaler.
    ratio_unsat = clamp(1f0 - (tmp / max_infil), 0f0, 1f0)
    
    ratio_powered = ratio_unsat > 0f0 ? ratio_unsat ^ b_i : 0f0
    as_val = 1f0 - ratio_powered
    ratio_beta = ratio_powered > 0f0 ? ratio_powered ^ (1f0 / b_i) : 0f0
    
    # 6. Series Expansion for exact curve integration
    dummy = 1f0
    ratio_pow_term = ratio_beta
    for k in 1:40
        dummy += (b_i * ratio_pow_term) / (b_i + Float32(k))
        ratio_pow_term *= ratio_beta
    end

    beta_asp = as_val + (1f0 - as_val) * (1f0 - ratio_beta) * dummy
    
    # 7. Final Calculation 
    # Apply the mathematically selected multiplier (beta_asp) directly 
    # unless full saturation naturally demands maximum potential evaporation.
    esoil = is_saturated ? pe : pe * beta_asp
    esoil = esoil * (1f0 - cov) * cv
    
    # 8. Cap at Available Moisture
    avail = max(sm_top - resid_top, 0f0)
    esoil = clamp(esoil, 0f0, avail)
    
    return esoil
end


function calculate_soil_evaporation!(
    soil_evap,
    soil_moisture, soil_moisture_max, potential_evaporation, 
    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture, AreaFract_gpu
)
    # Clear the output array first (since we accumulate into it)
    fill!(soil_evap, 0f0)

    # --- 2. Apply Logic (Accumulate over Veg Types and Bands) ---
    N_veg = size(cv_gpu, 4)
    N_bands = size(AreaFract_gpu, 3)
    for i in 1:N_veg
        for b in 1:N_bands
            # Esoil = Esoil_pot * beta(sm) * (1-fcanopy) * Cv per tile.
            # pe passed in is Step 2 (snow-blended) PE which captures snow energy effects.
            @views @. soil_evap += soil_evap_kernel(
                soil_moisture[:,:,1],           
                soil_moisture_max[:,:,1],       
                residual_moisture[:,:,1],       
                potential_evaporation[:,:,b,i],
                b_infilt_gpu,                   
                cv_gpu[:,:,1,i] * AreaFract_gpu[:,:,b],
                coverage_gpu[:,:,1,i]           
            )
        end
    end
    
    return nothing
end


function update_soil!(model)
    (;
        moisture, evaporation, surface_runoff, subsurface_runoff, 
        saturated_fraction, infiltration, interlayer_drainage
    ) = model.soil_variables
    (;
        nijssen_infilt_b, residual_moisture, maximum_moisture,
        hydraulic_conductivity, campbell_n, 
        nijssen_lin_reservoir, nijssen_nolin_reservoir,
        moisture_depth_baseflow_transition, baseflow_curve_exp
    ) = model.soil_parameters
    (; soil_potential_evaporation) = model.surface_energy_variables
    (; vegetation_fraction, canopy_coverage) = model.vegetation_parameters
    (; snow_band_area_fraction) = model.grid_parameters
    (; throughfall, transpiration_layers) = model.canopy_variables

    calculate_soil_evaporation!(
        evaporation, moisture, maximum_moisture, soil_potential_evaporation,  # Step 2 (snow-blended) PE
        nijssen_infilt_b, vegetation_fraction, canopy_coverage, residual_moisture, snow_band_area_fraction
    )

    calculate_surface_runoff!(
        surface_runoff, saturated_fraction,
        throughfall, moisture,
        maximum_moisture, nijssen_infilt_b, vegetation_fraction, snow_band_area_fraction
    )

    calculate_infiltration!(infiltration, throughfall, surface_runoff, vegetation_fraction)

    # Soil moisture update
    transpiration_grid = sum(transpiration_layers .* vegetation_fraction, dims=4)
    solve_runoff_and_drainage!(
        moisture, subsurface_runoff, surface_runoff, interlayer_drainage,
        infiltration, evaporation, transpiration_grid,
        maximum_moisture, hydraulic_conductivity, residual_moisture, campbell_n,
        nijssen_nolin_reservoir, nijssen_lin_reservoir, 
        moisture_depth_baseflow_transition, baseflow_curve_exp
    )
    return nothing
end
