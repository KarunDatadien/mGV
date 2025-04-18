function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)

    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions
    Kdry_min = (0.135 .* bulk_dens_min .+ 64.7) ./ (soil_dens_min .- 0.947 .* bulk_dens_min)
    Kdry = (1 .- organic_frac) .* Kdry_min .+ organic_frac .* Kdry_org

 #   println("soil_dens_min: ", minimum(soil_dens_min), " / ", maximum(soil_dens_min))
 #   println("bulk_dens_min: ", minimum(bulk_dens_min), " / ", maximum(bulk_dens_min))

 #   println("Kdry_min: ", minimum(Kdry_min), " / ", maximum(Kdry_min))
 #   println("Kdry_org: ", minimum(Kdry_org), " / ", maximum(Kdry_org))
 #   println("Kdry: ", minimum(Kdry), " / ", maximum(Kdry))

    # Fractional degree of saturation
    Sr = ifelse.(porosity .> 0, moist ./ porosity, 0.0)

    # Compute Ks of mineral soil based on quartz content
    Ks_min = ifelse.((quartz .< 0.2) .& (quartz .<= 1.0),
                    7.7 .^ quartz .* 3.0 .^ (1.0 .- quartz),
                    ifelse.(quartz .<= 1.0,
                            7.7 .^ quartz .* 2.2 .^ (1.0 .- quartz),
                            0.0))

    Ks = (1 .- organic_frac) .* Ks_min .+ organic_frac .* Ks_org
 #   println("CHECK! organic_frac: ", minimum(organic_frac), " / ", maximum(organic_frac))
 #   println("Ks_min: ", minimum(Ks_min), " / ", maximum(Ks_min))
#    println("Ks_org: ", minimum(Ks_org), " / ", maximum(Ks_org))

    # Calculate Ksat depending on whether the soil is unfrozen (Wu == moist) or partially frozen
    Ksat = ifelse.(Wu .== moist,
                  Ks .^ (1.0 .- porosity) .* Kw .^ porosity,
                  Ks .^ (1.0 .- porosity) .* Ki .^ (porosity .- Wu) .* Kw .^ Wu)

#    println("Ksat: ", minimum(Ksat), " / ", maximum(Ksat))

    # Compute the effective saturation parameter, Ke
    Ke = ifelse.(Wu .== moist,
                0.7 .* log10.(max.(Sr, 1e-10)) .+ 1.0,
                Sr)

 #   println("Ke: ", minimum(Ke), " / ", maximum(Ke))

    # Final Kappa calculation using ifelse to handle moist > 0 condition
    Kappa = ifelse.(moist .> 0,
                   max.((Ksat .- Kdry) .* Ke .+ Kdry, Kdry),
                   Kdry)


 #   println("Kappa: ", minimum(Kappa), " / ", maximum(Kappa))

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

function calculate_gsm_inv(soil_moisture::CuArray, soil_moisture_critical::CuArray, wilting_point::CuArray)
    # Initialize gsm_inv to zeros (handles full stress case: soil_moisture < wilting_point)
    println("soil_moisture shape: ", size(soil_moisture))

    gsm_inv = CUDA.zeros(eltype(soil_moisture), size(soil_moisture,1), size(soil_moisture,2), size(soil_moisture,3) )
    println("gsm_inv shape: ", size(gsm_inv))
    
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


function update_topsoil_moisture(
    prec_gpu, soil_moisture_old, soil_moisture_max, surface_runoff, Q_12, soil_evaporation, depth_gpu
)
    # Sum the soil moisture and maximum soil moisture across the top two layers (layer 1)
    topsoil_moisture = sum(soil_moisture_old[:, :, 1:2], dims=3)  # W_1^-[N+1]
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:2], dims=3)  # W_1^c

    # Compute updated soil moisture (W_1^+[N+1]) using Eq. 19
    # W_1^+[N+1] = W_1^-[N+1] + (P - Q_d[N+1] - Q_12[N+1] - E_1) * Δt
    # Here, throughfall is P * Δt, surface_runoff is Q_d * Δt, Q_12 is Q_12 * Δt, soil_evaporation is E_1 * Δt
    # Assuming Δt = 1
    topsoil_moisture_new = topsoil_moisture .+ (prec_gpu .- surface_runoff .- Q_12 .- soil_evaporation)

    # Ensure soil moisture stays within bounds (0 to topsoil_moisture_max)
    topsoil_moisture_new = max.(0.0, min.(topsoil_moisture_max, topsoil_moisture_new))

    # Since soil_moisture_old has size (204, 180, 3), we need to update the first two layers
    # and preserve the third layer (layer 2 in VIC model, to be updated separately)
    soil_moisture_new = copy(soil_moisture_old)

    # Distribute topsoil_moisture_new back to layers 1 and 2 based on depth ratios
    # Total depth of layer 1 (top two layers)
    total_depth = sum(depth_gpu[:, :, 1:2], dims=3)
    # Fraction of total depth for each sub-layer
    fraction_layer1 = depth_gpu[:, :, 1:1] ./ total_depth
    fraction_layer2 = depth_gpu[:, :, 2:2] ./ total_depth
    # Distribute moisture according to depth ratios
    soil_moisture_new[:, :, 1:1] = topsoil_moisture_new .* fraction_layer1
    soil_moisture_new[:, :, 2:2] = topsoil_moisture_new .* fraction_layer2

    return soil_moisture_new
end

function calculate_drainage_Q12(
    soil_moisture_old, soil_moisture_max, ksat_gpu, resid_moist, B_p
)
    # Initialize Q_12 contributions from each sub-layer
    Q_12_total = CUDA.zeros(204, 180, 1)

    # Compute drainage for each sub-layer in layer 1 (layers 1 and 2 in the array)
    for layer in 1:2
        # Use the soil moisture, K_s, theta_r, and B_p for the current sub-layer
        sublayer_moisture = soil_moisture_old[:, :, layer:layer]  # Size (204, 180, 1)
        sublayer_moisture_max = soil_moisture_max[:, :, layer:layer]  # Size (204, 180, 1)
        K_s = ksat_gpu[:, :, layer:layer]  # Size (204, 180, 1)
        theta_r = resid_moist[:, :, layer:layer]  # Size (204, 180, 1)
        B_p_sublayer = B_p[:, :, layer:layer]  # Size (204, 180, 1)

        # Compute drainage for this sub-layer
        drainage_ratio = (sublayer_moisture .- theta_r) ./ (sublayer_moisture_max .- theta_r)
        drainage_ratio = max.(drainage_ratio, 0.0)  # Ensure non-negative
        Q_12_sublayer = K_s .* drainage_ratio .^ ((2.0 ./ B_p_sublayer) .+ 3.0)
        Q_12_sublayer = max.(Q_12_sublayer, 0.0)

        # Add to the total drainage
        Q_12_total .+= Q_12_sublayer
    end

    return Q_12_total
end

function update_bottomsoil_moisture(
    soil_moisture_old, soil_moisture_max, Q_b, Q_12
)
    # Bottom layer is the third layer (index 3) in your 3-layer setup
    bottomsoil_moisture = soil_moisture_old[:, :, 3:3]    # W_2^-[N+1], shape (204, 180, 1)
    bottomsoil_moisture_max = soil_moisture_max[:, :, 3:3]  # W_2^c, shape (204, 180, 1)

    # Eq. 22a: Compute new bottom soil moisture (W_2^+[N+1])
    # W_2^+[N+1] = W_2^-[N+1] + (Q_12 - Q_b - E_2) * Δt, with Δt = 1 day and E_2 = 0
    bottomsoil_moisture_new = bottomsoil_moisture .+ (Q_12 .- Q_b)

    # Ensure non-negative moisture
    bottomsoil_moisture_new = max.(0.0, min.(bottomsoil_moisture_max, bottomsoil_moisture_new))

    # Eq. 22b: Check for excess moisture and compute Q_b'' (Q_b^*[N+1])
    # If W_2^+[N+1] > W_2^c, cap moisture and add excess to runoff
    Q_b_excess = ifelse.(
        bottomsoil_moisture_new .> bottomsoil_moisture_max,
        bottomsoil_moisture_new .- bottomsoil_moisture_max,  # Q_b'' = excess moisture
        0.0
    )

    # Cap the new moisture at max (Eq. 22b)
    bottomsoil_moisture_new = ifelse.(
        bottomsoil_moisture_new .> bottomsoil_moisture_max,
        bottomsoil_moisture_max,
        bottomsoil_moisture_new
    )

    # Total subsurface runoff: Q_b + Q_b'' (Q_b^*[N+1] in paper)
    subsurface_runoff_total = Q_b .+ Q_b_excess

    # Update the full soil_moisture_old array with the new bottom layer values
    soil_moisture_old[:, :, 3:3] = bottomsoil_moisture_new

    return soil_moisture_old[:, :, 3:3], subsurface_runoff_total
end