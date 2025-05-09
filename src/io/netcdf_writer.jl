function create_output_netcdf(output_file::String, reference_array, reference_array2)
    println("Creating NetCDF output file...")
    out_ds = NCDataset(output_file, "c")
    
    # Define dimensions based on the reference array’s shape
    defDim(out_ds, "lon",   size(reference_array, 1))
    defDim(out_ds, "lat",   size(reference_array, 2))
    defDim(out_ds, "time",  size(reference_array, 3))
    defDim(out_ds, "nveg",  size(reference_array2, 4))
    defDim(out_ds, "layer", 3)
    defDim(out_ds, "top_layer", 1)

    # Define the output variables to be written
    precipitation_output = defVar(out_ds, "precipitation_output", Float32, ("lon", "lat", "time"),
                       deflatelevel = 0, # Compression done afterwards with compress_file_async
                       chunksizes   = (64, 64, 1))
    precipitation_output.attrib["units"]       = "mm/day"
    precipitation_output.attrib["description"] = "Daily precipitation"

    water_storage_output = defVar(out_ds, "water_storage_output", Float32, ("lon", "lat", "time", "nveg"))
    water_storage_output.attrib["units"] = "mm"
    water_storage_output.attrib["description"] = "Water stored in the canopy per vegetation"
    
    water_storage_summed_output = defVar(out_ds, "water_storage_summed_output", Float32, ("lon", "lat", "time"))
    water_storage_summed_output.attrib["units"] = "mm"
    water_storage_summed_output.attrib["description"] = "Total water stored in the canopy"

    Q12_output = defVar(out_ds, "Q12_output", Float32, ("lon", "lat", "time", "nveg"))
    Q12_output.attrib["units"] = "mm"
    Q12_output.attrib["description"] = "Drainage from layer 1 to layer 2 per vegetation"
    
    Q12_summed_output = defVar(out_ds, "Q12_summed_output", Float32, ("lon", "lat", "time"))
    Q12_summed_output.attrib["units"] = "mm"
    Q12_summed_output.attrib["description"] = "Total drainage from layer 1 to layer 2"

    tair_output = defVar(out_ds, "tair_output", Float32, ("lon", "lat", "time"))
    tair_output.attrib["units"] = "°C"
    tair_output.attrib["description"] = "Air temperature at reference height"

    tsurf_output = defVar(out_ds, "tsurf_output", Float32, ("lon", "lat", "time"))
    tsurf_output.attrib["units"] = "°C"
    tsurf_output.attrib["description"] = "Surface temperature per vegetation"
    
#    tsurf_summed_output = defVar(out_ds, "tsurf_summed_output", Float32, ("lon", "lat", "time"))
#    tsurf_summed_output.attrib["units"] = "°C"
#    tsurf_summed_output.attrib["description"] = "Summed surface temperature"
    
    canopy_evaporation_output = defVar(out_ds, "canopy_evaporation_output", Float32, ("lon", "lat", "time", "nveg"))
    canopy_evaporation_output.attrib["units"] = "mm"
    canopy_evaporation_output.attrib["description"] = "Evaporation from canopy interception per vegetation"
    
    canopy_evaporation_summed_output = defVar(out_ds, "canopy_evaporation_summed_output", Float32, ("lon", "lat", "time"))
    canopy_evaporation_summed_output.attrib["units"] = "mm"
    canopy_evaporation_summed_output.attrib["description"] = "Total evaporation from canopy interception"

    transpiration_output = defVar(out_ds, "transpiration_output", Float32, ("lon", "lat", "time", "nveg"))
    transpiration_output.attrib["units"] = "mm"
    transpiration_output.attrib["description"] = "Plant transpiration per vegetation"
    
    transpiration_summed_output = defVar(out_ds, "transpiration_summed_output", Float32, ("lon", "lat", "time"))
    transpiration_summed_output.attrib["units"] = "mm"
    transpiration_summed_output.attrib["description"] = "Total plant transpiration"

    aerodynamic_resistance_output = defVar(out_ds, "aerodynamic_resistance_output", Float32, ("lon", "lat", "time", "nveg"))
    aerodynamic_resistance_output.attrib["units"] = "s/m"
    aerodynamic_resistance_output.attrib["description"] = "Aerodynamic resistance per vegetation"    

    aerodynamic_resistance_summed_output = defVar(out_ds, "aerodynamic_resistance_summed_output", Float32, ("lon", "lat", "time"))
    aerodynamic_resistance_summed_output.attrib["units"] = "s/m"
    aerodynamic_resistance_summed_output.attrib["description"] = "Total aerodynamic resistance"    

    potential_evaporation_output = defVar(out_ds, "potential_evaporation_output", Float32, ("lon", "lat", "time", "nveg"))
    potential_evaporation_output.attrib["units"] = "mm"
    potential_evaporation_output.attrib["description"] = "Potential evaporation per vegetation"

    potential_evaporation_summed_output = defVar(out_ds, "potential_evaporation_summed_output", Float32, ("lon", "lat", "time"))
    potential_evaporation_summed_output.attrib["units"] = "mm"
    potential_evaporation_summed_output.attrib["description"] = "Potential evaporation"

    net_radiation_output = defVar(out_ds, "net_radiation_output", Float32, ("lon", "lat", "time", "nveg"))
    net_radiation_output.attrib["units"] = "W/m^2"
    net_radiation_output.attrib["description"] = "Net radiation, per vegetation"

    net_radiation_summed_output = defVar(out_ds, "net_radiation_summed_output", Float32, ("lon", "lat", "time"))
    net_radiation_summed_output.attrib["units"] = "W/m^2"
    net_radiation_summed_output.attrib["description"] = "Net radiation"

    max_water_storage_output = defVar(out_ds, "max_water_storage_output", Float32, ("lon", "lat", "time", "nveg"))
    max_water_storage_output.attrib["units"] = "mm"
    max_water_storage_output.attrib["description"] = "The maximum amount of water intercepted by the canopy per vegetation"

    max_water_storage_summed_output = defVar(out_ds, "max_water_storage_summed_output", Float32, ("lon", "lat", "time"))
    max_water_storage_summed_output.attrib["units"] = "mm"
    max_water_storage_summed_output.attrib["description"] = "The maximum amount of water intercepted by the canopy"
    
    soil_evaporation_output = defVar(out_ds, "soil_evaporation_output", Float32, ("lon", "lat", "time", "top_layer"))
    soil_evaporation_output.attrib["units"] = "mm"
    soil_evaporation_output.attrib["description"] = "Evaporation from the soil surface per top soil layer"
    
    soil_temperature_output = defVar(out_ds, "soil_temperature_output", Float32, ("lon", "lat", "time", "layer"))
    soil_temperature_output.attrib["units"] = "°C"
    soil_temperature_output.attrib["description"] = "Soil temperature per layer"
    
    soil_moisture_output = defVar(out_ds, "soil_moisture_output", Float32, ("lon", "lat", "time", "layer"))
    soil_moisture_output.attrib["units"] = "kg/m^3"
    soil_moisture_output.attrib["description"] = "Volumetric soil moisture content per layer"

    total_et_output = defVar(out_ds, "total_et_output", Float32, ("lon", "lat", "time"))
    total_et_output.attrib["units"] = "mm"
    total_et_output.attrib["description"] = "Total evapotranspiration, weighted sum of canopy evaporation, transpiration, and bare soil evaporation across all cover classes"
    
    total_runoff_output = defVar(out_ds, "total_runoff_output", Float32, ("lon", "lat", "time"))
    total_runoff_output.attrib["units"] = "mm"
    total_runoff_output.attrib["description"] = "Total runoff, weighted sum of surface and subsurface runoff across all cover classes"

    kappa_array_output = defVar(out_ds, "kappa_array_output", Float32, ("lon", "lat", "time", "layer"))
    cs_array_output = defVar(out_ds, "cs_array_output", Float32, ("lon", "lat", "time", "layer"))

    return out_ds, precipitation_output, water_storage_output, water_storage_summed_output, Q12_output, Q12_summed_output,
           tair_output, tsurf_output, canopy_evaporation_output,
           canopy_evaporation_summed_output, transpiration_output, transpiration_summed_output, aerodynamic_resistance_output, aerodynamic_resistance_summed_output,
           potential_evaporation_output, potential_evaporation_summed_output, net_radiation_output,
           net_radiation_summed_output, max_water_storage_output, max_water_storage_summed_output,
           soil_evaporation_output, soil_temperature_output, soil_moisture_output,  total_et_output, total_runoff_output,
           kappa_array_output, cs_array_output

end
