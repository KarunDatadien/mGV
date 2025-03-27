function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, soil_density, bulk_density, organic_frac, porosity)

    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions
    Kdry_min = (0.135 .* bulk_dens_min .+ 64.7) ./ (soil_dens_min .- 0.947 .* bulk_dens_min)
    Kdry = (1 .- organic_frac) .* Kdry_min .+ organic_frac .* Kdry_org

    # Fractional degree of saturation
    Sr = ifelse.(porosity .> 0, moist ./ porosity, 0.0)

    # Compute Ks of mineral soil based on quartz content
    Ks_min = ifelse.(quartz .< 0.2,
                    7.7 .^ quartz .* 3.0 .^ (1.0 .- quartz),
                    7.7 .^ quartz .* 2.2 .^ (1.0 .- quartz))
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
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray, soil_moisture_critical::CuArray, wilting_point::CuArray, 
    root_gpu::CuArray
)
    # Compute soil moisture stress factors for both layers
    g_sm1 = calculate_gsm_inv(soil_moisture_old[:,:,2], soil_moisture_critical[:,:,2], wilting_point[:,:,2])
    g_sm2 = calculate_gsm_inv(soil_moisture_old[:,:,3], soil_moisture_critical[:,:,3], wilting_point[:,:,3])

    println("soil_moisture_old shape: ", size(soil_moisture_old))
    println("soil_moisture_critical shape: ", size(soil_moisture_critical))
    println("wilting_point shape: ", size(wilting_point))

    # Compute transpiration for each layer
    factor = (1 .- (water_storage ./ max_water_storage) .^ (2/3)) .* potential_evaporation

    println("Size of factor: ", size(factor))
    println("Size of aerodynamic_resistance: ", size(aerodynamic_resistance))
    println("Size of rarc_gpu: ", size(rarc_gpu))
    println("Size of canopy_resistance: ", size(canopy_resistance))
    println("Size of g_sm1: ", size(g_sm1))
    println("Size of root_gpu: ", size(root_gpu))


    E_t1 = factor .* (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc_gpu .+ canopy_resistance ./ g_sm1))
    E_t2 = factor .* (aerodynamic_resistance ./ (aerodynamic_resistance .+ rarc_gpu .+ canopy_resistance ./ g_sm2))

    # Calculate the general case (weighted transpiration) as the base
    E_t = root_gpu[:,:,2,:] .* E_t1 .+ root_gpu[:,:,3,:] .* E_t2

    # Define conditions for Case 1 and Case 2
    case1 = (root_gpu[:,:,3,:] .>= 0.5) .& (soil_moisture_old[:,:,3] .>= soil_moisture_critical[:,:,3])
    case2 = (root_gpu[:,:,2,:] .>= 0.5) .& (soil_moisture_old[:,:,2] .>= soil_moisture_critical[:,:,2])

    # Apply Case 1: Override with E_t2 where case1 is true
    E_t = ifelse.(case1, E_t2, E_t)

    # Apply Case 2: Override with E_t1 where case2 is true AND case1 is false (to preserve priority)
    E_t = ifelse.(case2 .& .!case1, E_t1, E_t)

    return E_t
end