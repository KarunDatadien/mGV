include("packages.jl")
include("utils.jl")
include("init.jl")
include("io.jl")
include("evapotranspiration.jl")

# Loop over years and process precipitation data
for year in start_year:end_year
    println("============ Start run for year: $year ============")

    println("Loading files and allocating memory...")

    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")

    (prec,    prec_cpu_preload,    prec_gpu)    = read_and_allocate_conditionally(input_prec_prefix,    year, prec_var)
    (tair,    tair_cpu_preload,    tair_gpu)    = read_and_allocate_conditionally(input_tair_prefix,    year, tair_var)
    (wind,    wind_cpu_preload,    wind_gpu)    = read_and_allocate_conditionally(input_wind_prefix,    year, wind_var)
    (vp,      vp_cpu_preload,      vp_gpu)      = read_and_allocate_conditionally(input_vp_prefix,      year, vp_var)
    (swdown,  swdown_cpu_preload,  swdown_gpu)  = read_and_allocate_conditionally(input_swdown_prefix,  year, swdown_var)
    (lwdown,  lwdown_cpu_preload,  lwdown_gpu)  = read_and_allocate_conditionally(input_lwdown_prefix,  year, lwdown_var)
    
    println("Open output file...")
    @time out_ds, prec_scaled, tair_scaled = create_output_netcdf(output_file, prec)

    # Set number of days in the current year
    num_days = size(prec, 3)

    println("Running...")
    # Process and write data one day at a time
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        # Explicitly copy the cleaned data to the GPU
        if GPU_USE == true
            CUDA.copyto!(prec_gpu, prec_cpu_preload[:, :, day])
            CUDA.copyto!(tair_gpu, tair_cpu_preload[:, :, day])
            CUDA.copyto!(wind_gpu, wind_cpu_preload[:, :, day])
            CUDA.copyto!(vp_gpu, vp_cpu_preload[:, :, day]) 
            CUDA.copyto!(swdown_gpu, swdown_cpu_preload[:, :, day])
            CUDA.copyto!(lwdown_gpu, lwdown_cpu_preload[:, :, day]) 

            apply_gpu_transformations!(prec_gpu, tair_gpu, 50)

            # Write results to the NetCDF file from the GPU
            prec_scaled[:, :, day] = Array(prec_gpu)
            tair_scaled[:, :, day] = Array(tair_gpu)
        else
            apply_cpu_transformations!(prec_cpu_preload[:, :, day], tair_cpu_preload[:, :, day], 50)
            prec_scaled[:, :, day] = prec_cpu_preload[:, :, day]
            tair_scaled[:, :, day] = tair_cpu_preload[:, :, day]
        end
    end

    println("Close output file...")
    close(out_ds)

    println("============ Completed run for year: $year ============\n")

    println("Postprocessing for year $year:")
    compress_file_async(output_file, 1)    

end

println("Done!")
