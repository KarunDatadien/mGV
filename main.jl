include("packages.jl")
include("init.jl")
include("io.jl")
include("utils.jl")
include("evapotranspiration.jl")

# Loop over years and process precipitation data
for year in start_year:end_year
    println("============ Start run for year: $year ============")

    println("Loading files and allocating memory...")

    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")

    # Read input variables and allocate GPU arrays
    @time (d_prec,    prec,      prec_gpu)           = read_and_allocate(input_prec_prefix,    year, prec_var)
    (d_tair,    tair,      tair_gpu)           = read_and_allocate(input_tair_prefix,    year, tair_var)
    (d_wind,    wind,      wind_gpu)           = read_and_allocate(input_wind_prefix,    year, wind_var)
    (d_vp,      vp,        vp_gpu)             = read_and_allocate(input_vp_prefix,      year, vp_var)
    (d_swdown,  swdown,    swdown_gpu)         = read_and_allocate(input_swdown_prefix,  year, swdown_var)
    (d_lwdown,  lwdown,    lwdown_gpu)         = read_and_allocate(input_lwdown_prefix,  year, lwdown_var)
 
    println("Load precipitation input...")
    @time prec_cpu_preload = prec[:, :, :]
    println("Load air temperature input...")
    @time tair_cpu_preload = tair[:, :, :]

    println("Open output file...")
    @time out_ds, prec_scaled, tair_scaled = create_output_netcdf(output_file, prec)

    # Set number of days in the current year
    num_days = size(prec, 3)

    println("Running...")
    # Process and write data one day at a time
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        # Explicitly copy the cleaned data to the GPU
        CUDA.copyto!(prec_gpu, prec_cpu_preload[:, :, day])
        CUDA.copyto!(tair_gpu, tair_cpu_preload[:, :, day]) 

        apply_gpu_transformations!(prec_gpu, tair_gpu, 50)

        # Write results directly to the NetCDF file from the GPU
        prec_scaled[:, :, day] = Array(prec_gpu)
        tair_scaled[:, :, day] = Array(tair_gpu)
    end

    println("Close output file...")
    close(out_ds)

    println("============ Completed run for year: $year ============\n")

    println("Postprocessing for year $year:")
    compress_file_async(output_file, 1)    

end

println("Done!")
