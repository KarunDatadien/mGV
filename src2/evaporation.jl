function canopy_evap_physics(
    ws, max_ws, pot_evap, ra, rarc, 
    prec, lai, tair, elev, rmin
)

    # --- 1. Physics Calculations ---
    # Using your existing project functions
    slope = calculate_svp_slope(tair)
    latent_heat = calculate_latent_heat(tair)

    scale_height = calculate_scale_height(tair, elev)
    surface_pressure = 101325.0f0 * exp(-elev / scale_height)
    gamma_val = 1628.6f0 * surface_pressure / latent_heat

    # --- 2. Resistances ---
    rc = rmin / max(lai, 1f-6)
    inv_ra = 1f0 / max(ra, 1f-6)

    # --- 3. Denominators (Penman-Monteith) ---
    den_w = slope + gamma_val * (1f0 + rarc * inv_ra)
    den_rc = den_w + (gamma_val * rc * inv_ra)

    E_p_wet = pot_evap * (den_rc / max(den_w, 1f-6))

    # --- 4. VIC Evaporative Scaling (Branchless) ---
    # The original VIC model uses conditional logic to evaluate if the canopy is fully wet or dry.
    # We replace that with a continuous mathematical formulation:
    # 
    #   Wratio: Evaluates the current canopy water storage as a fraction of its maximum capacity.
    #           It is safely clamped between 0.0 (completely dry) and 1.0 (fully wet).
    #
    #   ra_ratio: Computes the aerodynamic resistance modifier.
    # 
    #   canopy_evap_star: Calculates the potential canopy evaporation scaled non-linearly 
    #                     (to the 2/3rds power) based on how much water is currently present.
    Wratio = clamp(ws / max(max_ws, 1f-6), 0f0, 1f0)
    ra_ratio = ra / max(ra + rarc, 1f-6)
    
    canopy_evap_star = (Wratio ^ (2f0 / 3f0)) * E_p_wet * ra_ratio

    # --- 5. Fraction Calculation (f_n) ---
    # Determines what fraction of the timestep the canopy will be evaporating.
    # If the available water (storage + new precipitation) is less than the theoretical 
    # evaporation amount (canopy_evap_star), the canopy will dry out partway through the timestep (f_n_val < 1.0).
    f_n_val = clamp((ws + prec) / max(canopy_evap_star, 1f-6), 0f0, 1f0)

    if canopy_evap_star <= 1f-6
        f_n_val = 1f0
    end

    # --- 6. Final Evaporation ---
    # The physical evaporation that occurs during the wet fraction of the timestep.
    evap = f_n_val * canopy_evap_star
    
    # Sanitize outputs to prevent NaN propagation
    if isnan(evap) | abs(evap > 1f15)
        evap = 0f0
    end

    return evap, f_n_val
end

function calculate_max_water_storage(lai, canopy_coverage, fillvalue_threshold)
    (; K_L) = Constants

    max_water_storage = ifelse(canopy_coverage > 1.0f-5, (K_L * lai) / canopy_coverage, 0f0)
    max_water_storage = ifelse(
        isnan(max_water_storage) | (abs(max_water_storage) > fillvalue_threshold), 
        0f0, 
        max_water_storage
    )
    return max_water_storage
end

@kernel function canopy_evaporation_kernel!(
    lai,
    canopy_coverage,
    fillvalue_threshold,
    maximum_water_storage,
    canopy_evaporation,
    wet_fraction,
    water_storage,
    potential_evaporation,
    aerodynamic_resistance,
    architectural_resistance,
    precipitation,
    air_temperature,
    elevation,
    minimum_resistance
)
    I = @index(Global)

    maximum_water_storage[I] = calculate_max_water_storage(lai[I], canopy_coverage[I], fillvalue_threshold)

    canopy_evaporation[I], wet_fraction[I] = canopy_evap_physics(
        water_storage[I],
        maximum_water_storage[I],
        potential_evaporation[I],
        aerodynamic_resistance[I],
        architectural_resistance[I],
        precipitation[I],
        lai[I],
        air_temperature[I],
        elevation[I],
        minimum_resistance[I]
    )
end

function update_canopy_evaporation!(model::Model)
    (; air_temperature, precipitation) = model.forcing_variables
    (; elevation) = model.grid_parameters
    (; potential_evaporation, aerodynamic_resistance) = model.surface_energy_variables
    (; canopy_evaporation, wet_fraction, water_storage, maximum_water_storage) = model.canopy_variables
    (; lai, canopy_coverage, minimum_resistance, architectural_resistance) = model.vegetation_parameters
    current_month = month(model.clock.time)

    # Some arrays are (nx, ny, 1, veg) and some are (nx, ny)
    #  all arrays need the same shape (nx, ny, nbands, nveg) for the GPU kernel
    nbands = size(maximum_water_storage, 3)
    nveg = size(maximum_water_storage, 4)

    canopy_evaporation_kernel!(device_backend)(
        repeat(@view(lai[:,:,[current_month],:]), outer=[1,1,nbands]),
        repeat(@view(canopy_coverage[:,:,[current_month],:]), outer=[1,1,nbands]),
        model.config.fillvalue_threshold,
        maximum_water_storage,
        canopy_evaporation,
        wet_fraction,
        water_storage,
        potential_evaporation,
        aerodynamic_resistance,
        repeat(architectural_resistance, outer=[1,1,nbands]),
        repeat(precipitation, outer=[1,1,nbands,nveg]),
        repeat(air_temperature, outer=[1,1,nbands,nveg]),
        repeat(elevation, outer=[1,1,nbands,nveg]),
        repeat(minimum_resistance, outer=[1,1,nbands]),
        ndrange = length(canopy_evaporation)
    )

    # 3. Post-Process: Zero out bare soil (last index)
    @view(canopy_evaporation[:, :, :, end]) .= 0f0
    return nothing
end

