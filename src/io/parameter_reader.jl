function read_and_allocate_parameter(varname::String)
    println("Loading $varname parameter input...")

    # 1) Open netCDF file, read variable into a CPU array and copy array into preload
    dataset       = NetCDF.open(input_param_file)
    cpu_arr       = dataset[varname]
    var_dims      = size(dataset[varname])  # Get the dimensions of the variable

    # Handle slicing based on dimensionality
    if length(var_dims) == 2
        cpu_preload = dataset[varname][:, :]
        println("Element type for 2D: ", eltype(cpu_preload))
    elseif length(var_dims) == 3
        cpu_preload = dataset[varname][:, :, :]
        println("Element type for 3D: ", eltype(cpu_preload))
    elseif length(var_dims) == 4
        cpu_preload = dataset[varname][:, :, :, :]
        println("Element type for 4D: ", eltype(cpu_preload))
    else
        error("Unsupported variable dimensionality: ", length(var_dims))
    end

    # Print array dimension sizes
    full_size = size(dataset[varname])
    println("Full size of $varname: ", full_size)

    # 2) Conditionally allocate a GPU array
    if GPU_USE
        # Adjust dimensions for the GPU array
        adjusted_dims = if length(var_dims) == 4
            (var_dims[1], var_dims[2], (var_dims[3] == 12 ? 1 : var_dims[3]), var_dims[4])  # Third dimension is reduced to 1 if time dimension (i.e. months)
        else
            var_dims  # For 2D or 3D cases, keep original dimensions
        end

        gpu_arr = CUDA.zeros(float_type, adjusted_dims...)  # Allocate based on adjusted dimensions
        println("Allocated GPU array of size: ", size(gpu_arr))
        return cpu_preload, gpu_arr
    else
        return cpu_preload, nothing
    end
end

function read_and_allocate_forcing(prefix::String, year::Int, varname::String)
    println("Loading $varname forcing input...")

    # 1) Open netCDF file, read variable into a CPU array and copy array into preload
    file_path     = "$(prefix)$(year).nc"
    dataset       = NetCDF.open(file_path)
    cpu_arr       = dataset[varname]
    cpu_preload   = dataset[varname][:, :, :]
    
    # 2) Conditionally allocate a GPU array
    if GPU_USE
        gpu_arr = CUDA.zeros(float_type, size(cpu_arr, 1), size(cpu_arr, 2))
        println("Allocated GPU array of size: ", size(gpu_arr))
        return cpu_preload, gpu_arr
    else
        return cpu_preload, nothing
    end
end

function gpu_load_static_inputs(cpu_vars, gpu_vars)
    for (cpu, gpu) in zip(cpu_vars, gpu_vars)
        CUDA.copyto!(gpu, cpu)
    end
end

function gpu_load_monthly_inputs(month, month_prev, cpu_vars, gpu_vars)
    if month != month_prev
        for (cpu, gpu) in zip(cpu_vars, gpu_vars)
            CUDA.copyto!(gpu, cpu[:, :, month, :])
        end
    end
end

function gpu_load_daily_inputs(day, day_prev, cpu_vars, gpu_vars)
    if day != day_prev
        for (cpu, gpu) in zip(cpu_vars, gpu_vars)
            # println("CPU Array Type: ", eltype(cpu))
            # println("GPU Array Type: ", eltype(gpu))
            CUDA.copyto!(gpu, cpu[:, :, day])
        end
    end
end
