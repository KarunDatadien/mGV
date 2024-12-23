# Check the output directory exists, otherwise create it
if !isdir(output_dir)
    println("Output directory '$output_dir' does not exist. Creating it...")
    mkpath(output_dir)
end

# Helper function to read a NetCDF variable
function read_netcdf_variable(prefix, year, variable_name)
    file_path = "$(prefix)$(year).nc"
    dataset = NCDataset(file_path, "r")
    return dataset, dataset[variable_name]  # Return both the dataset and the variable
end

# Helper function: reads a single netCDF variable and allocates a GPU array
function read_and_allocate(prefix, year, varname)
    d, arr = read_netcdf_variable(prefix, year, varname)
    return d, arr, CUDA.zeros(Float32, size(arr,1), size(arr,2))
end

function create_output_netcdf(output_file::String, reference_array)
    # Create a new NetCDF file to store the scaled data
    out_ds = NCDataset(output_file, "c")
    
    # Define dimensions based on the reference array’s shape
    defDim(out_ds, "lon",   size(reference_array, 1))
    defDim(out_ds, "lat",   size(reference_array, 2))
    defDim(out_ds, "time",  size(reference_array, 3))
    defDim(out_ds, "layer", 3)

    # Define the variables to be written
    pr_scaled = defVar(out_ds, "scaled_precipitation", Float32, ("lon", "lat", "time"),
                       deflatelevel = 0, # Compression done afterwards with compress_file_async
                       chunksizes   = (512, 512, 1),
                       fillvalue    = -9999.0f0)

    tair_scaled = defVar(out_ds, "scaled_tair", Float32, ("lon", "lat", "time"),
                         deflatelevel = 0, # Compression done afterwards with compress_file_async
                         chunksizes   = (512, 512, 1),
                         fillvalue    = -9999.0f0)

    # Set attributes                     
    pr_scaled.attrib["units"]       = "mm/day"
    pr_scaled.attrib["description"] = "Daily precipitation scaled with GPU computations (optimized)"
                     

    return out_ds, pr_scaled, tair_scaled
end
