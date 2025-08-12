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
function calculate_interlayer_drainage(Ksat, current_moisture, max_moisture, residual_moisture, expt)
    drainage_ratio = (current_moisture .- residual_moisture) ./ (max_moisture .- residual_moisture)
    drainage_ratio = clamp.(drainage_ratio, 0.0f0, 1.0f0)
    Q12 = Ksat .* drainage_ratio .^ expt
    return clamp.(Q12, 0.0f0, Inf32)
end

#function calculate_interlayer_drainage(
#    Ksat,
#    current_moisture,
#    residual_moisture,
#    max_moisture,
#    expt
#)
#    # Epsilon for numerical stability
#    EPSILON = float_type(1e-9)
#
#    # Effective moisture above residual
#    eff_moist = max.(current_moisture .- residual_moisture, 0.0f0)
#    max_eff_moist = max.(max_moisture .- residual_moisture, EPSILON)
#
#    # Term 1: (initial effective moisture)^{1 - expt}
#    term1 = (eff_moist .+ EPSILON) .^ (1.0f0 .- expt)
#
#    # Drainage potential over the time step: Ksat / (max_eff_moist ^ expt) * (1 - expt)
#drainage_potential = (Ksat ./ (max_eff_moist .^ expt)) .* (expt .- 1.0f0)
#
#    # Inner power base: term1 - drainage_potential
#    inner_pow_base = term1 .- drainage_potential
#
#    # Clamp inner_pow_base to prevent negative base in power
#    inner_pow_base = max.(inner_pow_base, 0.0f0)
#
#    # New effective moisture after drainage: (inner_pow_base)^{1 / (1 - expt)}
#    # Handle expt ==1 special case (linear drainage)
#    inv_denom = 1.0f0 ./ (1.0f0 .- expt .+ EPSILON)
#    final_eff_moist = inner_pow_base .^ inv_denom
#
#    # Total drainage: initial eff_moist - final_eff_moist
#    Q12 = eff_moist .- final_eff_moist
#
#    # Ensure drainage is non-negative and doesn't exceed available eff_moist
#    Q12 = clamp.(Q12, 0.0f0, eff_moist)
#
#    return clamp.(Q12, 0.0f0, Inf32)
#end


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
    soil_moisture_max,   # Max moisture (porosity) (3D array)
    ksat_gpu,            # Saturated hydraulic conductivity (3D array)
    residual_moisture,   # Residual moisture (3D array)
    expt_gpu,            # Brooks-Corey exponent (3D array)
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu # These are 2D arrays
)
    # Convert all inputs to Float32 for GPU compatibility
    surface_inflow_f32 = Float32.(surface_inflow)
    soil_evaporation_f32 = Float32.(soil_evaporation)
    transpiration_f32 = Float32.(transpiration)
    soil_moisture_old_f32 = Float32.(soil_moisture_old)
    soil_moisture_max_f32 = Float32.(soil_moisture_max)
    ksat_gpu_f32 = Float32.(ksat_gpu)
    residual_moisture_f32 = Float32.(residual_moisture)
    expt_gpu_f32 = Float32.(expt_gpu)
    Dsmax_gpu_f32 = Float32.(Dsmax_gpu)
    Ds_gpu_f32 = Float32.(Ds_gpu)
    Ws_gpu_f32 = Float32.(Ws_gpu)
    c_expt_gpu_f32 = Float32.(c_expt_gpu)

    num_layers = size(soil_moisture_old_f32, 3)

    # 1. Calculate drainage flux between all upper layers (1 to N-1)
    Q12 = CUDA.zeros(Float32, size(surface_inflow_f32, 1), size(surface_inflow_f32, 2), num_layers - 1)
    for l in 1:(num_layers - 1)
        Q12[:, :, l] = calculate_interlayer_drainage(
            ksat_gpu_f32[:, :, l], soil_moisture_old_f32[:, :, l],
            soil_moisture_max_f32[:, :, l], residual_moisture_f32[:, :, l], expt_gpu_f32[:, :, l]
        )
    end

    # 2. Calculate baseflow exiting the bottom layer (N)
    baseflow = calculate_baseflow(
        soil_moisture_old_f32[:, :, num_layers], residual_moisture_f32[:, :, num_layers],
        soil_moisture_max_f32[:, :, num_layers], Dsmax_gpu_f32, Ds_gpu_f32, Ws_gpu_f32, c_expt_gpu_f32
    )

    # Aggregate fluxes across all vegetation types, padding for layers
    transp_layers = size(transpiration_f32, 3)
    total_transpiration_per_layer = CUDA.zeros(Float32, size(soil_moisture_old_f32))
    total_transpiration_per_layer[:, :, 1:transp_layers] = sum(transpiration_f32, dims=4)[:, :, 1:transp_layers, 1]

    total_soil_evaporation_per_layer = CUDA.zeros(Float32, size(soil_moisture_old_f32))
    total_soil_evaporation_per_layer[:, :, 1] = sum(soil_evaporation_f32, dims=4)[:, :, 1, 1]

    # Initialize new soil moisture
    soil_moisture_new = copy(soil_moisture_old_f32)

    if num_layers == 1
        soil_moisture_new[:, :, 1] .+= surface_inflow_f32 .- (
            total_soil_evaporation_per_layer[:, :, 1] .+
            total_transpiration_per_layer[:, :, 1] .+
            baseflow
        )
    else
        soil_moisture_new[:, :, 1] .+= surface_inflow_f32 .- (
            total_soil_evaporation_per_layer[:, :, 1] .+
            total_transpiration_per_layer[:, :, 1] .+
            Q12[:, :, 1]
        )

        for l in 2:(num_layers - 1)
            soil_moisture_new[:, :, l] .+= Q12[:, :, l-1] .- (
                total_transpiration_per_layer[:, :, l] .+
                Q12[:, :, l]
            )
        end

        soil_moisture_new[:, :, num_layers] .+= Q12[:, :, num_layers - 1] .- (
            total_transpiration_per_layer[:, :, num_layers] .+
            baseflow
        )
    end

    soil_moisture_new = min.(soil_moisture_new, soil_moisture_max_f32)
    soil_moisture_new = max.(residual_moisture_f32, soil_moisture_new)

    # Diagnostic: Check water balance, replacing NaNs with 0
    total_inflow = ifelse.(isnan.(surface_inflow_f32), 0.0f0, surface_inflow_f32)
    total_outflow = ifelse.(isnan.(sum(total_soil_evaporation_per_layer, dims=3)), 0.0f0, sum(total_soil_evaporation_per_layer, dims=3)) .+
                    ifelse.(isnan.(sum(total_transpiration_per_layer, dims=3)), 0.0f0, sum(total_transpiration_per_layer, dims=3)) .+
                    ifelse.(isnan.(sum(Q12, dims=3)), 0.0f0, sum(Q12, dims=3)) .+
                    ifelse.(isnan.(baseflow), 0.0f0, baseflow)
    water_balance_error = total_inflow .- total_outflow .- (soil_moisture_new .- soil_moisture_old_f32)
    water_balance_error = ifelse.(isnan.(water_balance_error), 0.0f0, water_balance_error)
    println("Max water balance error: ", maximum(abs.(water_balance_error)))

    return soil_moisture_new, baseflow, Q12
end
