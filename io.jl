# Check the output directory exists, otherwise create it
if !isdir(output_dir)
    println("Output directory '$output_dir' does not exist. Creating it...")
    mkpath(output_dir)
end

function read_and_allocate(prefix::String, year::Int, varname::String)
    println("Loading $varname input...")

    # 1) Open netCDF file, read variable into a CPU array and copy array into preload
    if endswith(prefix, ".nc")
        file_path = prefix
    else
        file_path = "$(prefix)$(year).nc"
    end
    dataset       = NetCDF.open(file_path)
    cpu_arr       = dataset[varname]
    var_dims      = size(dataset[varname])  # Get the dimensions of the variable

    # Handle slicing based on dimensionality
    if length(var_dims) == 2
        cpu_preload = dataset[varname][:, :]
    elseif length(var_dims) == 3
        cpu_preload = dataset[varname][:, :, :]
    elseif length(var_dims) == 4
        cpu_preload = dataset[varname][:, :, :, :]
    else
        error("Unsupported variable dimensionality: ", length(var_dims))
    end

    # Print array dimension sizes
    full_size = size(dataset[varname])
    println("Full size of $varname: ", full_size)

    # 2) Conditionally allocate a GPU array
    if GPU_USE
        # Adjust dimensions for the GPU array
        adjusted_dims = if length(var_dims) == 3
            (var_dims[1], var_dims[2], 1)  # Third dimension is reduced to 1
        elseif length(var_dims) == 4
            (var_dims[1], var_dims[2], 1, var_dims[4])  # Third dimension is reduced to 1
        else
            var_dims  # For 2D or other cases, keep original dimensions
        end

        gpu_arr = CUDA.zeros(Float32, adjusted_dims...)  # Allocate based on adjusted dimensions
        println("Allocated GPU array of size: ", size(gpu_arr))
        return cpu_arr, cpu_preload, gpu_arr
    else
        return cpu_arr, cpu_preload, nothing
    end
end

function create_output_netcdf(output_file::String, reference_array)
    println("Creating NetCDF output file...")
    out_ds = NCDataset(output_file, "c")
    
    # Define dimensions based on the reference array’s shape
    defDim(out_ds, "lon",   size(reference_array, 1))
    defDim(out_ds, "lat",   size(reference_array, 2))
    defDim(out_ds, "time",  size(reference_array, 3))
    defDim(out_ds, "layer", 3)

    # Define the variables to be written
    pr_scaled = defVar(out_ds, "scaled_precipitation", Float32, ("lon", "lat", "time"),
                       deflatelevel = 0, # Compression done afterwards with compress_file_async
                       chunksizes   = (64, 64, 1),
                       fillvalue    = -9999.0f0)

    tair_scaled = defVar(out_ds, "scaled_tair", Float32, ("lon", "lat", "time"),
                         deflatelevel = 0, # Compression done afterwards with compress_file_async
                         chunksizes   = (64, 64, 1),
                         fillvalue    = -9999.0f0)

    # Set attributes                     
    pr_scaled.attrib["units"]       = "mm/day"
    pr_scaled.attrib["description"] = "Daily precipitation scaled with GPU computations (optimized)"
                     

    return out_ds, pr_scaled, tair_scaled
end
