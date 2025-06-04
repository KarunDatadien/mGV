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


function update_topsoil_moisture(prec_gpu, throughfall, soil_moisture_old, soil_moisture_max, surface_runoff, Q_12, soil_evaporation, depth_gpu, E_1_t)
    # Sum the soil moisture and maximum soil moisture across the top two layers (layer 1)
#    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:2], dims=3)  # W_1^c


    # Compute updated soil moisture for vegetated layers (n = 1 to nveg) using throughfall
    topsoil_moisture_new_veg = (sum_with_nan_handling(throughfall[:, :, :, 1:end], 4) )  .- sum_with_nan_handling(E_1_t[:, :, :, 1:end-1], 4) .- surface_runoff[:, :, :, end:end] .- soil_evaporation

    # Compute updated soil moisture for bare soil layer (n = nveg+1) using prec_gpu
#    topsoil_moisture_new_bare = (prec_gpu .* cv_gpu[:, :, :, end:end] .- surface_runoff[:, :, :, end:end] .- soil_evaporation)

    # Combine updated soil moistures
#    topsoil_moisture_new = topsoil_moisture_new_veg .+ topsoil_moisture_new_bare .- Q_12
    topsoil_moisture_new = topsoil_moisture_new_veg .- Q_12

    # Since soil_moisture_old has size (204, 180, 3, nveg+1), update the first two layers
    soil_moisture_new = copy(soil_moisture_old)

    # Distribute topsoil_moisture_new back to layers 1 and 2 based on depth ratios
    total_depth = sum(depth_gpu[:, :, 1:2], dims=3)
    fraction_layer1 = depth_gpu[:, :, 1:1] ./ total_depth
    fraction_layer2 = depth_gpu[:, :, 2:2] ./ total_depth


    println("topsoil_moisture_new_veg shape: ", size(topsoil_moisture_new_veg))
  #  println("topsoil_moisture_new_bare shape: ", size(topsoil_moisture_new_bare))
    println("sum_with_nan_handling(surface_runoff[:, :, :, 1:end-1], 4)  shape: ", size(sum_with_nan_handling(surface_runoff[:, :, :, 1:end-1], 4) ))
    println("Q_12 shape: ", size(Q_12))



    # Apply depth ratios to all layers (n = 1 to nveg+1)
    soil_moisture_new[:, :, 1:1] = soil_moisture_old[:, :, 1:1] .+ topsoil_moisture_new .* fraction_layer1
    soil_moisture_new[:, :, 2:2] = soil_moisture_old[:, :, 2:2] .+ topsoil_moisture_new .* fraction_layer2

    soil_moisture_new[:, :, 1:1] = max.(0.0, min.(soil_moisture_max[:, :, 1:1], soil_moisture_new[:, :, 1:1]))
    soil_moisture_new[:, :, 2:2] = max.(0.0, min.(soil_moisture_max[:, :, 2:2], soil_moisture_new[:, :, 2:2]))
 


    return soil_moisture_new
end



function calculate_drainage_Q12(soil_moisture_old, soil_moisture_max, ksat_gpu, residual_moisture, expt_gpu)
    # TODO: check if this function is correct, gives different results from VIC-c

    # Compute drainage for each sub-layer (layers 1 and 2) for all n
    sublayer_moisture = soil_moisture_old[:, :, 1:2]
    sublayer_moisture_max = soil_moisture_max[:, :, 1:2] 
    K_s = ksat_gpu[:, :, 1:2] # [mm/day]
    theta_r = residual_moisture[:, :, 1:2] 
    expt_sublayer = expt_gpu[:, :, 1:2]  

    # Compute the drainage ratio for each sub-layer: (W_1[n] - theta_r) / (W_1^c - theta_r)
    drainage_ratio = (sublayer_moisture .- theta_r) ./ (sublayer_moisture_max .- theta_r)
    drainage_ratio = max.(drainage_ratio, 0.0)  # Ensure non-negative

    # Compute Q_12 for each sub-layer using Eq. 20: K_s * (drainage_ratio)^((2/B_p) + 3)
    Q_12_sublayer = K_s .* drainage_ratio .^ (expt_sublayer)
    Q_12_sublayer = max.(Q_12_sublayer, 0.0)

    # Sum the contributions from the two sub-layers for each n
    Q_12 = sum(Q_12_sublayer, dims=3) 

    return Q_12
end


function update_bottomsoil_moisture(soil_moisture_new, soil_moisture_max, Q_b, Q_12, E_2_t)
    # Bottom layer is the third layer (index 3) in your 3-layer setup, for all n
    bottomsoil_moisture = soil_moisture_new[:, :, 3:3]    # W_2^-[n]
    bottomsoil_moisture_max = soil_moisture_max[:, :, 3:3]  # W_2^c
    
    # Eq. 22a: Compute new bottom soil moisture (W_2^+[n])
    # For vegetated layers (n = 1 to nveg-1), include E_2^t[n]; for bare soil (nveg), E_2 = 0
    bottomsoil_moisture_new = bottomsoil_moisture .+ Q_12 .- Q_b .- sum_with_nan_handling(E_2_t[:, :, :, 1:end-1], 4) 

    # Ensure non-negative moisture
    bottomsoil_moisture_new = max.(0.0, min.(bottomsoil_moisture_max, bottomsoil_moisture_new))

    # Eq. 22b: Check for excess moisture and compute Q_b'' (Q_b^*[n])
    # If W_2^+[n] > W_2^c, cap moisture and add excess to runoff
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

    # Total subsurface runoff: Q_b + Q_b'' (Q_b^*[n] in paper)
    subsurface_runoff_total = Q_b .+ Q_b_excess

    # Update the full soil_moisture_old array with the new bottom layer values
    soil_moisture_new[:, :, 3:3] = bottomsoil_moisture_new

    return soil_moisture_new[:, :, 3:3], subsurface_runoff_total
end
