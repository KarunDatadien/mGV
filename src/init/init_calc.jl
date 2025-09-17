function calculate_soil_properties(bulk_dens_gpu, soil_dens_gpu, depth_gpu, Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu)

    # === Calculate Bulk Density and Porosity ===
    bulk_dens_min = (bulk_dens_gpu .- organic_frac * bulk_dens_org) ./ (1 .- organic_frac)
    soil_dens_min = (soil_dens_gpu .- organic_frac * soil_dens_org) ./ (1 .- organic_frac)

    bulk_density = (1 .- organic_frac) .* bulk_dens_min .+ organic_frac .* bulk_dens_org
    soil_density = (1 .- organic_frac) .* soil_dens_min .+ organic_frac .* soil_dens_org

    porosity = 1 .- bulk_density ./ soil_density
    porosity = max.(porosity, 0.0)


    # === Calculate Maximum Soil Moisture ===
    soil_moisture_max = depth_gpu .* porosity .* 1000

    # === Field Capacity, Wilting Point, and Critical Moisture ===
    soil_moisture_critical = Wcr_gpu .* soil_moisture_max
    field_capacity = Wfc_gpu .* soil_moisture_max
    wilting_point = Wpwp_gpu .* soil_moisture_max
    residual_moisture = residmoist_gpu .* depth_gpu .* 1000

    return bulk_dens_min, soil_dens_min, porosity, soil_moisture_max, soil_moisture_critical, field_capacity, wilting_point, residual_moisture
end