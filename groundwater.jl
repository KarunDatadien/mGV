function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, soil_density, bulk_density, organic_frac, porosity)
    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions
    Kdry_min = (0.135 .* bulk_dens_min .+ 64.7) ./ (soil_dens_min .- 0.947 .* bulk_dens_min)
    Kdry     = (1 .- organic_frac) .* Kdry_min .+ organic_frac .* Kdry_org

    # Initialize Kappa with the same shape as moist, pre-filled with Kdry
    Kappa = fill(Kdry, size(moist))

    # Process only where moisture content is greater than 0
    mask = moist .> 0
    if any(mask)

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

        # Update Kappa for locations with moisture
        Kappa[mask] .= (Ksat[mask] .- Kdry) .* Ke[mask] .+ Kdry
        Kappa[mask] .= max.(Kappa[mask], Kdry)
    end

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
