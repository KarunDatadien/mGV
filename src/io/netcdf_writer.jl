function create_output_netcdf(output_file::String, reference_array, reference_array2)
    println("Creating NetCDF output file...")
    out_ds = NCDataset(output_file, "c")
    
    # Define dimensions based on the reference array’s shape
    defDim(out_ds, "lon",   size(reference_array, 1))
    defDim(out_ds, "lat",   size(reference_array, 2))
    defDim(out_ds, "time",  size(reference_array, 3))
    defDim(out_ds, "nveg",  size(reference_array2, 4))
    defDim(out_ds, "layer", 3)

    # Define the output variables to be written
    pr_scaled = defVar(out_ds, "scaled_precipitation", Float32, ("lon", "lat", "time"),
                       deflatelevel = 0, # Compression done afterwards with compress_file_async
                       chunksizes   = (64, 64, 1))

    water_storage_output = defVar(out_ds, "water_storage_output", Float32, ("lon", "lat", "time", "nveg"))

    Q12_output = defVar(out_ds, "Q12_output", Float32, ("lon", "lat", "time"))

    tair_output = defVar(out_ds, "tair_output", Float32, ("lon", "lat", "time"))

    tsurf_output = defVar(out_ds, "tsurf_output", Float32, ("lon", "lat", "time", "layer", "nveg"))

    canopy_evaporation_output = defVar(out_ds, "canopy_evaporation_output", Float32, ("lon", "lat", "time", "nveg"))

    transpiration_output = defVar(out_ds, "transpiration_output", Float32, ("lon", "lat", "time", "nveg"))

    aerodynamic_resistance_output = defVar(out_ds, "aerodynamic_resistance_output", Float32, ("lon", "lat", "time", "nveg"))

    # Set attributes                     
    pr_scaled.attrib["units"]       = "mm/day"
    pr_scaled.attrib["description"] = "Daily precipitation scaled with GPU computations (optimized)"

    return out_ds, pr_scaled, water_storage_output, Q12_output, tair_output, tsurf_output, canopy_evaporation_output, transpiration_output, aerodynamic_resistance_output
end