function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)
    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions
    Kdry_min = (0.135 .* bulk_dens_min .+ 64.7) ./ (soil_dens_min .- 0.947 .* bulk_dens_min)
    Kdry = (1 .- organic_frac) .* Kdry_min .+ organic_frac .* Kdry_org

    # Fractional degree of saturation
    Sr = ifelse.(porosity .> 0, moist ./ porosity, 0.0)

    # Compute Ks of mineral soil based on quartz content
    Ks_min = ifelse.((quartz .< 0.2) .& (quartz .<= 1.0),
                    7.7 .^ quartz .* 3.0 .^ (1.0 .- quartz),
                    ifelse.(quartz .<= 1.0,
                            7.7 .^ quartz .* 2.2 .^ (1.0 .- quartz),
                            0.0))

    Ks = (1 .- organic_frac) .* Ks_min .+ organic_frac .* Ks_org

    # Calculate Ksat depending on whether the soil is unfrozen (Wu == moist) or partially frozen
    Ksat = ifelse.(Wu .== moist,
                  Ks .^ (1.0 .- porosity) .* Kw .^ porosity,
                  Ks .^ (1.0 .- porosity) .* Ki .^ (porosity .- Wu) .* Kw .^ Wu)

    # Compute the effective saturation parameter, Ke
    Ke = ifelse.(Wu .== moist,
                0.7 .* log10.(max.(Sr, 1e-10)) .+ 1.0,
                Sr)

    # Final Kappa calculation using ifelse to handle moist > 0 condition
    Kappa = ifelse.(moist .> 0,
                   max.((Ksat .- Kdry) .* Ke .+ Kdry, Kdry),
                   Kdry)

    return Kappa
end

function volumetric_heat_capacity(soil_fract, water_fract, ice_fract, organic_frac)
    # Constant values are volumetric heat capacities in J/m^3/K
    Cs = 2.0e6 .* soil_fract .* (1 .- organic_frac) .+
         2.7e6 .* soil_fract .* organic_frac .+
         4.2e6 .* water_fract .+
         1.9e6 .* ice_fract .+
         1.3e3 .* (1.0 .- (soil_fract .+ water_fract .+ ice_fract))  # Air component

    return Cs
end

function calculate_gsm_inv(soil_moisture, soil_moisture_critical, wilting_point)
    ## Initialize gsm_inv to zeros (handles full stress case: soil_moisture < wilting_point)
    # println("soil_moisture shape: ", size(soil_moisture))

    gsm_inv = CUDA.zeros(eltype(soil_moisture), size(soil_moisture,1), size(soil_moisture,2), size(soil_moisture,3) )
    # println("gsm_inv shape: ", size(gsm_inv))
    
    # Calculate the partial stress term for all elements
    partial_stress = (soil_moisture .- wilting_point) ./ (soil_moisture_critical .- wilting_point)

    # Use ifelse to handle the two remaining cases:
    # - Case 1: No stress (soil_moisture >= soil_moisture_critical) -> 1
    # - Case 2: Partial stress (wilting_point <= soil_moisture < soil_moisture_critical) -> partial_stress
    # - Case 3: Anything still zero is implicitly soil_moisture < wilting_point

    gsm_inv .= ifelse.(soil_moisture .>= soil_moisture_critical,
                      1.0,
                      partial_stress)

    return gsm_inv
end


"""
    calculate_interlayer_drainage(Ksat, current_moisture, max_moisture, Wfc, expt)

Calculates the gravitational drainage (flux) between two soil layers based on the
Brooks-Corey formulation used in the VIC model.
"""
function calculate_interlayer_drainage(Ksat, current_moisture, max_moisture, Wfc, expt)
    effective_moisture = current_moisture .- Wfc
    denominator = max_moisture .- Wfc
    EPSILON = 1.0f-9
    drainage_ratio = effective_moisture ./ (denominator .+ EPSILON)
    drainage_ratio = clamp.(drainage_ratio, 0.0f0, 1.0f0)
    Q12 = Ksat .* drainage_ratio .^ expt
    return clamp.(Q12, 0.0f0, Inf32)
end


"""
    calculate_baseflow(bottom_moisture, resid_moist, max_moist, Dsmax, Ds, Ws, c_expt)

Calculates baseflow from the bottom soil layer using the ARNO model.
"""
function calculate_baseflow(bottom_moisture, resid_moist, max_moist, Dsmax, Ds, Ws, c_expt)
    EPSILON = 1.0f-9
    rel_moist = (bottom_moisture .- resid_moist) ./ (max_moist .- resid_moist .+ EPSILON)
    rel_moist = clamp.(rel_moist, 0.0f0, 1.0f0)
    frac = Dsmax .* Ds ./ Ws
    baseflow = frac .* rel_moist
    is_above_ws = rel_moist .> Ws
    frac_nonlinear = (rel_moist .- Ws) ./ (1.0f0 .- Ws .+ EPSILON)
    baseflow .+= is_above_ws .* Dsmax .* (1.0f0 .- Ds ./ Ws) .* (frac_nonlinear .^ c_expt)
    return clamp.(baseflow, 0.0f0, Inf32)
end


"""
Solves the runoff and drainage for a multi-layered soil column.

This function calculates the movement of water between soil layers (drainage),
the runoff from the surface, and the baseflow from the bottom layer. It then
updates the soil moisture for each layer based on a water balance that now
correctly includes inflows (surface water, drainage from above) and outflows
(bare soil evaporation, plant transpiration, drainage to below, baseflow).

Args:
    surface_inflow: Water reaching the soil surface (2D array).
    soil_evaporation: Evaporation from bare soil in each layer (3D array).
                      NOTE: This is assumed to only have non-zero values in the first layer.
    transpiration: Water uptake by plants from each layer (3D array).
                   NOTE: This array dictates which layers lose water to transpiration.
    soil_moisture_old: Moisture from the previous time step (3D array).
    Wfc_gpu: Field capacity of each layer (3D array).
    soil_moisture_max: Maximum moisture (porosity) of each layer (3D array).
    ksat_gpu: Saturated hydraulic conductivity of each layer (3D array).
    residual_moisture: Residual moisture of each layer (3D array).
    expt_gpu: Brooks-Corey exponent for each layer (3D array).
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu: ARNO baseflow parameters (2D arrays).

Returns:
    A tuple containing:
    - soil_moisture_new: The updated soil moisture for each layer (3D array).
    - baseflow: The calculated baseflow from the bottom layer (2D array).
    - Q12: The drainage flux between layers (3D array).
"""
function solve_runoff_and_drainage(
    surface_inflow,      # Water reaching the soil surface (2D array)
    soil_evaporation,    # Evaporation from each layer, per veg type (4D array)
    transpiration,       # Transpiration from each layer, per veg type (4D array)
    soil_moisture_old,   # Moisture from previous step (3D array)
    Wfc_gpu,             # Field capacity (3D array)
    soil_moisture_max,   # Max moisture (porosity) (3D array)
    ksat_gpu,            # Saturated hydraulic conductivity (3D array)
    residual_moisture,   # Residual moisture (3D array)
    expt_gpu,            # Brooks-Corey exponent (3D array)
    # ARNO baseflow parameters for the bottom layer
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu # These are 2D arrays
)
    
    num_layers = size(soil_moisture_old, 3)
    
    # 1. Calculate drainage flux between all upper layers (1 to N-1)
    # Note: This will be an empty array if num_layers is 1.
    Q12 = CUDA.zeros(Float32, size(surface_inflow, 1), size(surface_inflow, 2), num_layers - 1)
    for l in 1:(num_layers - 1)
        Q12[:, :, l] = calculate_interlayer_drainage(
            ksat_gpu[:, :, l], soil_moisture_old[:, :, l],
            soil_moisture_max[:, :, l], Wfc_gpu[:, :, l], expt_gpu[:, :, l]
        )
    end
    
    # 2. Calculate baseflow exiting the bottom layer (N)
    baseflow = calculate_baseflow(
        soil_moisture_old[:, :, num_layers], residual_moisture[:, :, num_layers],
        soil_moisture_max[:, :, num_layers], Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu
    )
    
    # Aggregate fluxes across all vegetation types before subtraction
    total_transpiration_per_layer = sum(transpiration, dims=4)[:,:,:,1]
    total_soil_evaporation_per_layer = sum(soil_evaporation, dims=4)[:,:,:,1]

    # 3. Update soil moisture in a cascade from top to bottom
    soil_moisture_new = copy(soil_moisture_old)

    # --- FIX: Use a conditional structure and add bounds checks for robustness ---
    if num_layers == 1
        # --- Update for a single-layer model ---
        # Inflow is surface inflow. Outflow is evaporation, transpiration, and baseflow.
        outflow_l1 = total_soil_evaporation_per_layer[:, :, 1] .+ total_transpiration_per_layer[:, :, 1] .+ baseflow
        soil_moisture_new[:, :, 1] .+= surface_inflow .- outflow_l1
    
    else # num_layers >= 2
        # --- Update for a multi-layer model (2+ layers) ---
        # Update layer 1
        outflow_l1 = total_soil_evaporation_per_layer[:, :, 1] .+ total_transpiration_per_layer[:, :, 1] .+ Q12[:, :, 1]
        soil_moisture_new[:, :, 1] .+= surface_inflow .- outflow_l1

        # Update intermediate layers (this loop only runs if num_layers >= 3)
        for l in 2:(num_layers - 1)
            inflow_l = Q12[:, :, l-1]
            drainage_l = Q12[:, :, l]
            # Defensively check if transpiration data exists for this layer before subtracting
            if l <= size(total_transpiration_per_layer, 3)
                soil_moisture_new[:, :, l] .+= inflow_l .- total_transpiration_per_layer[:, :, l] .- drainage_l
            else
                soil_moisture_new[:, :, l] .+= inflow_l .- drainage_l
            end
        end

        # Update bottom layer
        inflow_bottom = Q12[:, :, num_layers - 1]
        # Defensively check if transpiration data exists for the bottom layer before subtracting
        if num_layers <= size(total_transpiration_per_layer, 3)
            soil_moisture_new[:, :, num_layers] .+= inflow_bottom .- total_transpiration_per_layer[:, :, num_layers] .- baseflow
        else
            soil_moisture_new[:, :, num_layers] .+= inflow_bottom .- baseflow
        end
    end
    
    # --- Final checks ---
    # Cap moisture at max and prevent it from falling below residual.
    soil_moisture_new = clamp.(soil_moisture_new, residual_moisture, soil_moisture_max)
    
    return soil_moisture_new, baseflow, Q12
end