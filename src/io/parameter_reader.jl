function read_and_allocate_parameter(varname::String)
    println("Loading $varname parameter input...")

    # 1) Open netCDF file, read variable into a CPU array and copy array into preload
    dataset       = NetCDF.open(input_param_file)
    cpu_arr       = dataset[varname]
    var_dims      = size(dataset[varname])  # Get the dimensions of the variable

    # Handle slicing based on dimensionality
    if length(var_dims) == 1
        cpu_preload = dataset[varname][:]
        println("Element type for 1D: ", eltype(cpu_preload))
    elseif length(var_dims) == 2
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

# This macro automates the creation of the CPU and GPU variables
macro load_params(vars...)
    quote
        # The `esc(...)` is what allows the macro to create variables
        # in the scope where you call it.
        $(map(vars) do var
            cpu_name = Symbol(String(var), "_cpu")
            gpu_name = Symbol(String(var), "_gpu")
            source_var = Symbol(String(var), "_var")
            
            # This generates the line: (var_cpu, var_gpu) = read_and_allocate_parameter(var_var)
            :(($(esc(cpu_name)), $(esc(gpu_name))) = read_and_allocate_parameter($(esc(source_var))))
        end...)
    end
end


"""
    @vars(names...)

Takes a list of base variable names and expands them into two lists:
one with a `_cpu` suffix and one with a `_gpu` suffix.
Returns a tuple containing the two lists.
"""
macro vars(names...)
    # Create a list of symbols with the `_cpu` suffix
    cpu_vars = [Symbol(String(name), "_cpu") for name in names]
    # Create a list of symbols with the `_gpu` suffix
    gpu_vars = [Symbol(String(name), "_gpu") for name in names]
    
    # The `esc()` is crucial for the macro to access the variables
    # from the scope where it is called.
    # We construct two array expressions, e.g., `[var1_cpu, var2_cpu]`
    # and `[var1_gpu, var2_gpu]`.
    quote
        ( [$(esc.(cpu_vars)...)], [$(esc.(gpu_vars)...)] )
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

"""
    @load_forcing(year_var, names...)

Takes a year variable and a list of base variable names. For each name,
it generates a call to `read_and_allocate_forcing`, creating the
corresponding `_cpu` and `_gpu` variables.
"""
macro load_forcing(year_var, names...)
    # The `quote ... end` block collects all the generated lines of code.
    quote
        # `map` iterates through each variable name provided (e.g., :prec, :tair)
        $(map(names) do name
            # Construct all the necessary variable names from the base name
            cpu_var      = esc(Symbol(String(name), "_cpu"))
            gpu_var      = esc(Symbol(String(name), "_gpu"))
            prefix_var   = esc(Symbol("input_", String(name), "_prefix"))
            source_var   = esc(Symbol(String(name), "_var"))
            year_esc     = esc(year_var)

            # This is the line of code that will be generated for each name:
            # e.g., (prec_cpu, prec_gpu) = read_and_allocate_forcing(input_prec_prefix, year, prec_var)
            :(($cpu_var, $gpu_var) = read_and_allocate_forcing($prefix_var, $year_esc, $source_var))
        end...)
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
