include("packages.jl")
include("init.jl")
include("io.jl")
include("utils.jl")
include("evapotranspiration.jl")

# Loop over years and process precipitation data
for year in start_year:end_year

    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")

    # Read input variables and allocate GPU arrays
    (d_prec,    prec,      prec_gpu)           = read_and_allocate(input_prec_prefix,    year, prec_var)
    (d_tair,    tair,      tair_gpu)           = read_and_allocate(input_tair_prefix,    year, tair_var)
    (d_wind,    wind,      wind_gpu)           = read_and_allocate(input_wind_prefix,    year, wind_var)
    (d_vp,      vp,        vp_gpu)             = read_and_allocate(input_vp_prefix,      year, vp_var)
    (d_swdown,  swdown,    swdown_gpu)         = read_and_allocate(input_swdown_prefix,  year, swdown_var)
    (d_lwdown,  lwdown,    lwdown_gpu)         = read_and_allocate(input_lwdown_prefix,  year, lwdown_var)
 
    # Clean and convert CPU data for all variables
    @time prec_cpu_cleaned = Float64.(replace(prec[:, :, :], missing => NaN))
    @time tair_cpu_cleaned          = Float64.(replace(tair[:, :, :], missing => NaN))

    # Create the output NetCDF (partly) using a reference array's shape
    out_ds, prec_scaled, tair_scaled = create_output_netcdf(output_file, prec)

    # Specify number of days in the current year
    num_days = size(prec, 3)

    # Process and write data one day at a time
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        # Explicitly copy the cleaned data to the GPU
        CUDA.copyto!(prec_gpu, prec_cpu_cleaned[:, :, day])
        CUDA.copyto!(tair_gpu, tair_cpu_cleaned[:, :, day]) 

        apply_gpu_transformations!(prec_gpu, tair_gpu, 50)

        # Write results directly to the NetCDF file from the GPU
        prec_scaled[:, :, day]   = Array(prec_gpu)
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


    println("Completed processing for year: $year\n")

    # Compress the output file to level 1 compression as a background process
    compress_file_async(output_file, 1)    
end

println("Done!")
