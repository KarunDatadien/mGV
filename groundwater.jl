function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, soil_density, bulk_density, organic_frac, porosity)

    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions
    Kdry_min = (0.135 .* bulk_dens_min .+ 64.7) ./ (soil_dens_min .- 0.947 .* bulk_dens_min)
    Kdry = (1 .- organic_frac) .* Kdry_min .+ organic_frac .* Kdry_org

    println("soil_dens_min: ", minimum(soil_dens_min), " / ", maximum(soil_dens_min))
    println("bulk_dens_min: ", minimum(bulk_dens_min), " / ", maximum(bulk_dens_min))


    println("Kdry_min: ", minimum(Kdry_min), " / ", maximum(Kdry_min))
    println("Kdry_org: ", minimum(Kdry_org), " / ", maximum(Kdry_org))
    println("Kdry: ", minimum(Kdry), " / ", maximum(Kdry))


    # Fractional degree of saturation
    Sr = ifelse.(porosity .> 0, moist ./ porosity, 0.0)

    # Compute Ks of mineral soil based on quartz content
    Ks_min = ifelse.((quartz .< 0.2) .& (quartz .<= 1.0),
                    7.7 .^ quartz .* 3.0 .^ (1.0 .- quartz),
                    ifelse.(quartz .<= 1.0,
                            7.7 .^ quartz .* 2.2 .^ (1.0 .- quartz),
                            0.0))

    Ks = (1 .- organic_frac) .* Ks_min .+ organic_frac .* Ks_org
    println("CHECK! organic_frac: ", minimum(organic_frac), " / ", maximum(organic_frac))
    println("Ks_min: ", minimum(Ks_min), " / ", maximum(Ks_min))
    println("Ks_org: ", minimum(Ks_org), " / ", maximum(Ks_org))

    # Calculate Ksat depending on whether the soil is unfrozen (Wu == moist) or partially frozen
    Ksat = ifelse.(Wu .== moist,
                  Ks .^ (1.0 .- porosity) .* Kw .^ porosity,
                  Ks .^ (1.0 .- porosity) .* Ki .^ (porosity .- Wu) .* Kw .^ Wu)

    println("Ksat: ", minimum(Ksat), " / ", maximum(Ksat))

    # Compute the effective saturation parameter, Ke
    Ke = ifelse.(Wu .== moist,
                0.7 .* log10.(max.(Sr, 1e-10)) .+ 1.0,
                Sr)

    println("Ke: ", minimum(Ke), " / ", maximum(Ke))

    # Final Kappa calculation using ifelse to handle moist > 0 condition
    Kappa = ifelse.(moist .> 0,
                   max.((Ksat .- Kdry) .* Ke .+ Kdry, Kdry),
                   Kdry)


    println("Kappa: ", minimum(Kappa), " / ", maximum(Kappa))

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

function calculate_transpiration(
    potential_evaporation::CuArray, aerodynamic_resistance::CuArray, rarc_gpu::CuArray, canopy_resistance::CuArray, 
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray, soil_moisture_critical::CuArray, 
    wilting_point::CuArray, root_gpu::CuArray
)
    println("soil_moisture_old shape: ", size(soil_moisture_old))
    println("soil_moisture_critical shape: ", size(soil_moisture_critical))
    println("wilting_point shape: ", size(wilting_point))
    println("root_gpu shape: ", size(root_gpu))

    # Compute stress factor
    g_sm = calculate_gsm_inv(soil_moisture_old, soil_moisture_critical, wilting_point)
    println("g_sm shape: ", size(g_sm))

    # Calculate modifier factor
    factor = (1 .- (water_storage ./ max_water_storage) .^ (2/3)) .* potential_evaporation
    println("factor shape: ", size(factor))

    # Expand for broadcasting
    g_sm_exp = reshape(g_sm, size(g_sm,1), size(g_sm,2), size(g_sm,3), 1)
    println("g_sm_exp shape: ", size(g_sm_exp))

    canopy_resistance_exp = reshape(canopy_resistance, size(canopy_resistance,1), size(canopy_resistance,2), 1, size(canopy_resistance,4))
    println("canopy_resistance_exp shape: ", size(canopy_resistance_exp))

    # Compute layer-wise transpiration
    E_layer = factor .* (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc_gpu .+ canopy_resistance_exp ./ g_sm_exp))
    println("E_layer shape: ", size(E_layer))

    # Weighted sum with root fraction
    E_t = sum(root_gpu .* E_layer, dims=3)
    println("E_t shape (final transpiration): ", size(E_t))

    return E_t
end
