include("init.jl")
include("io.jl")

# Loop over years and process precipitation data
for year in start_year:end_year
    output_file = joinpath(output_dir, "precipitation_scaled_$(year).nc")

    # Use helper function to read precipitation and tair data
    d_precip, precipitation = read_netcdf_variable(input_precip_prefix, year, "prec")
    d_tair, tair = read_netcdf_variable(input_tair_prefix, year, "tair")
    
    num_days = size(precipitation, 3)

    # Create a new NetCDF file to store the scaled data
    out_ds = NCDataset(output_file, "c")

    # Define dimensions and variables
    defDim(out_ds, "lon", size(precipitation, 1))
    defDim(out_ds, "lat", size(precipitation, 2))
    defDim(out_ds, "time", size(precipitation, 3))
    defDim(out_ds, "layer", 3)

    # Define a new variable for scaled precipitation
    pr_scaled = defVar(out_ds, "scaled_precipitation", Float32, ("lon", "lat", "time"),
                       deflatelevel=1, chunksizes=(512, 512, 1), fillvalue=-9999.0f0)
    tair_scaled = defVar(out_ds, "scaled_tair", Float32, ("lon", "lat", "time"),
                       deflatelevel=1, chunksizes=(512, 512, 1), fillvalue=-9999.0f0)

    precipitation_gpu = CUDA.zeros(Float32, size(precipitation, 1), size(precipitation, 2))
    tair_gpu = CUDA.zeros(Float32, size(tair, 1), size(tair, 2))

    # Step 1: Transfer the precipitation slice for the current day to the GPU
    @time precipitation_cpu_cleaned = Float32.(replace(precipitation[:, :, :], missing => NaN))
    @time tair_cpu_cleaned = Float32.(replace(tair[:, :, :], missing => NaN))

    # Process and write data one day at a time
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days

        # Step 2: Explicitly copy the cleaned data to the GPU
        CUDA.copyto!(precipitation_gpu, precipitation_cpu_cleaned[:,:,day])
        CUDA.copyto!(tair_gpu, tair_cpu_cleaned[:,:,day]) 

        for _ in 1:50
            precipitation_gpu .= sqrt.(max.(precipitation_gpu, 0) .+ 1)       # Square root transformation
            precipitation_gpu .= log.(precipitation_gpu .+ 1e-6)              # Logarithmic transformation
            precipitation_gpu .= exp.(precipitation_gpu) .* sin.(precipitation_gpu)      # Non-linear exp and sine combination
            tair_gpu .= tair_gpu .+ exp.(precipitation_gpu) .* sin.(tair_gpu) 
        end

        # Step 3: Write results directly to the NetCDF file from the GPU
        pr_scaled[:, :, day] = Array(precipitation_gpu)  # Transfer only final results back to CPU
        tair_scaled[:, :, day] = Array(tair_gpu)  # Transfer only final results back to CPU
    end

    # Add attributes to the variable
    pr_scaled.attrib["units"] = "mm/day"
    pr_scaled.attrib["description"] = "Daily precipitation scaled with GPU computations (optimized)"
    
    close(d_precip)
    close(d_tair)
    close(out_ds)

    println("Completed processing for year: $year\n")
end

println("Done!")
