include("packages.jl")
include("init.jl")
include("io.jl")
include("utils.jl")
include("evapotranspiration.jl")

# Loop over years and process precipitation data
for year in start_year:end_year

    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")

    # Read input variables and allocate GPU arrays
    (d_prec,    precipitation, precipitation_gpu)  = read_and_allocate(input_precip_prefix,  year, "prec")
    (d_tair,    tair,          tair_gpu)           = read_and_allocate(input_tair_prefix,    year, "tair")
    (d_wind,    wind,          wind_gpu)           = read_and_allocate(input_wind_prefix,    year, "wind")
    (d_vp,      vp,            vp_gpu)             = read_and_allocate(input_vp_prefix,      year, "vp")
    (d_swdown,  swdown,        swdown_gpu)         = read_and_allocate(input_swdown_prefix,  year, "swdown")
    (d_lwdown,  lwdown,        lwdown_gpu)         = read_and_allocate(input_lwdown_prefix,  year, "lwdown")
 
    # Clean and convert CPU data for all variables
    @time precipitation_cpu_cleaned = Float64.(replace(precipitation[:, :, :], missing => NaN))
    @time tair_cpu_cleaned          = Float64.(replace(tair[:, :, :],          missing => NaN))

    # Create the output NetCDF (partly) using a reference array's shape
    out_ds, pr_scaled, tair_scaled = create_output_netcdf(output_file, precipitation)

    # Specify number of days in the current year
    num_days = size(precipitation, 3)

    # Process and write data one day at a time
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        # Explicitly copy the cleaned data to the GPU
        CUDA.copyto!(precipitation_gpu, precipitation_cpu_cleaned[:, :, day])
        CUDA.copyto!(tair_gpu, tair_cpu_cleaned[:, :, day]) 

        apply_gpu_transformations!(precipitation_gpu, tair_gpu, 50)

        # Write results directly to the NetCDF file from the GPU
        pr_scaled[:, :, day]   = Array(precipitation_gpu)
        tair_scaled[:, :, day] = Array(tair_gpu)
    end

    # Close datasets
    close(d_prec)
    close(d_tair)
    close(d_wind)
    close(d_vp)
    close(d_swdown)
    close(d_lwdown)
    close(out_ds)

    # Compress the output file to level 1 compression
    compress_file_async(output_file, 1)
    
    println("Completed processing for year: $year\n")
end

println("Done!")
