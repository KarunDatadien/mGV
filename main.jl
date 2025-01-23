include("packages.jl")
include("constants.jl")
include("utils.jl")
include("init.jl")
include("io.jl")
include("physics.jl")
include("evapotranspiration.jl")

using .SimConstants

# Loop over years
for year in start_year:end_year
    println("============ Start run for year: $year ============")
    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")

    println("Loading parameter data and allocating memory...")
    (d0,    d0_cpu_preload,    d0_gpu)    = read_and_allocate(input_param_file,    year, d0_var)
    (z0,    z0_cpu_preload,    z0_gpu)    = read_and_allocate(input_param_file,    year, z0_var)
    
    println("Loading forcing data and allocating memory...")
    (prec,    prec_cpu_preload,    prec_gpu)    = read_and_allocate(input_prec_prefix,    year, prec_var)
    (tair,    tair_cpu_preload,    tair_gpu)    = read_and_allocate(input_tair_prefix,    year, tair_var)
    (wind,    wind_cpu_preload,    wind_gpu)    = read_and_allocate(input_wind_prefix,    year, wind_var)
    (vp,      vp_cpu_preload,      vp_gpu)      = read_and_allocate(input_vp_prefix,      year, vp_var)
    (swdown,  swdown_cpu_preload,  swdown_gpu)  = read_and_allocate(input_swdown_prefix,  year, swdown_var)
    (lwdown,  lwdown_cpu_preload,  lwdown_gpu)  = read_and_allocate(input_lwdown_prefix,  year, lwdown_var)

    out_ds, prec_scaled, tair_scaled = create_output_netcdf(output_file, prec)

    num_days = size(prec, 3) # Set number of days in the current year
    month_prev = 0

    println("Running...") 
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        month = day_to_month(day, year)
        #println("Day: $day, Month: $month")

        if GPU_USE == true
            # Explicitly copy preloaded data to the GPU
            # Load monthly input data
            if month != month_prev
                CUDA.copyto!(d0_gpu, d0_cpu_preload[:, :, month, :])
                CUDA.copyto!(z0_gpu, z0_cpu_preload[:, :, month, :])
            end
            # Load daily input data
            CUDA.copyto!(prec_gpu, prec_cpu_preload[:, :, day])
            CUDA.copyto!(tair_gpu, tair_cpu_preload[:, :, day])
            CUDA.copyto!(wind_gpu, wind_cpu_preload[:, :, day])
            CUDA.copyto!(vp_gpu, vp_cpu_preload[:, :, day]) 
            CUDA.copyto!(swdown_gpu, swdown_cpu_preload[:, :, day])
            CUDA.copyto!(lwdown_gpu, lwdown_cpu_preload[:, :, day]) 

            # Run computations
            aerodynamic_resistance = compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu)
            
            # Write results to the NetCDF file from the GPU
            prec_scaled[:, :, day] = Array(prec_gpu)
            tair_scaled[:, :, day] = Array(tair_gpu)
        else
            # CPU version (to be tested):
            # apply_cpu_transformations!(prec_cpu_preload[:, :, day], tair_cpu_preload[:, :, day], 50)
            # prec_scaled[:, :, day] = prec_cpu_preload[:, :, day]
            # tair_scaled[:, :, day] = tair_cpu_preload[:, :, day]
        end

        month_prev = month

    end

    println("Close output file...")
    close(out_ds)

    println("============ Completed run for year: $year ============\n")

    println("Postprocessing for year $year:")
    compress_file_async(output_file, 1)    

end

println("Done!")
