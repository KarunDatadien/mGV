include("packages.jl")
include("constants.jl")
include("utils.jl")
include("io.jl")
include("init.jl")
include("physics.jl")
include("evapotranspiration.jl")

println("Loading parameter data and allocating memory...")
using .SimConstants
@time begin
    (d0_cpu,        d0_gpu)        = read_and_allocate_parameter(d0_var)
    (z0_cpu,        z0_gpu)        = read_and_allocate_parameter(z0_var)
    (LAI_cpu,       LAI_gpu)       = read_and_allocate_parameter(LAI_var)
    (albedo_cpu,    albedo_gpu)    = read_and_allocate_parameter(albedo_var) 
    (rmin_cpu,      rmin_gpu)      = read_and_allocate_parameter(rmin_var) 
    (rarc_cpu,      rarc_gpu)      = read_and_allocate_parameter(rarc_var) 
    (elev_cpu,      elev_gpu)      = read_and_allocate_parameter(elev_var) 
    (root_cpu,      root_gpu)      = read_and_allocate_parameter(root_var)
    (Wcr_cpu,       Wcr_gpu)       = read_and_allocate_parameter(Wcr_var)
    (Wfc_cpu,       Wfc_gpu)       = read_and_allocate_parameter(Wfc_var)
    (Wpwp_cpu,      Wpwp_gpu)      = read_and_allocate_parameter(Wpwp_var)
    (depth_cpu,     depth_gpu)     = read_and_allocate_parameter(depth_var)
    (bulk_dens_cpu, bulk_dens_gpu) = read_and_allocate_parameter(bulk_dens_var)
    (soil_dens_cpu, soil_dens_gpu) = read_and_allocate_parameter(soil_dens_var)
end

# Loop over years
for year in start_year:end_year
    println("============ Start run for year: $year ============")
    println("Loading forcing data and allocating memory...")
    @time begin
        (prec_cpu,   prec_gpu)    = read_and_allocate_forcing(input_prec_prefix,    year, prec_var)
        (tair_cpu,   tair_gpu)    = read_and_allocate_forcing(input_tair_prefix,    year, tair_var)
        (wind_cpu,   wind_gpu)    = read_and_allocate_forcing(input_wind_prefix,    year, wind_var)
        (vp_cpu,     vp_gpu)      = read_and_allocate_forcing(input_vp_prefix,      year, vp_var)
        (swdown_cpu, swdown_gpu)  = read_and_allocate_forcing(input_swdown_prefix,  year, swdown_var)
        (lwdown_cpu, lwdown_gpu)  = read_and_allocate_forcing(input_lwdown_prefix,  year, lwdown_var)
    end

    println("Opening output file...")
    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")
    @time out_ds, prec_scaled, tair_scaled = create_output_netcdf(output_file, prec_cpu)

    println("Running...") 
    num_days = size(prec_cpu, 3) # Set number of days in the current year
    day_prev = 0
    month_prev = 0 # Initialize previous month to 0 at start of year
    
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        month = day_to_month(day, year)

        # Explicitly copy preloaded data to the GPU
        if GPU_USE == true

            gpu_load_static_inputs([rmin_cpu, rarc_cpu, elev_cpu], 
                                         [rmin_gpu, rarc_gpu, elev_gpu])

            gpu_load_monthly_inputs(month, month_prev, 
                                          [d0_cpu, z0_cpu, LAI_cpu, albedo_cpu], 
                                          [d0_gpu, z0_gpu, LAI_gpu, albedo_gpu])

            gpu_load_daily_inputs(day, day_prev, 
                                        [prec_cpu, tair_cpu, wind_cpu, vp_cpu, swdown_cpu, lwdown_cpu], 
                                        [prec_gpu, tair_gpu, wind_gpu, vp_gpu, swdown_gpu, lwdown_gpu])

            #println("Start computations for potential evaporation...") 
            tsurf = tair_gpu #TODO: calculate actual tsurf

           # aerodynamic_resistance = compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu)
           # canopy_resistance = compute_canopy_resistance(rmin_gpu, LAI_gpu)
           # net_radiation = calculate_net_radiation(swdown_gpu, lwdown_gpu, albedo_gpu, tsurf) 
           # potential_evaporation = calculate_potential_evaporation(tair_gpu, vp_gpu, elev_gpu, net_radiation, aerodynamic_resistance, canopy_resistance, rarc_gpu)
            
    #        canopy_evaporation = calculate_canopy_evaporation(W_i, LAI_gpu, potential_evaporation, aerodynamic_resistance, rarc_gpu, prec_gpu)
    #        transpiration = calculate_transpiration(potential_evaporation, aerodynamic_resistance, rarc_gpu, canopy_resistance, W_i, W_im, soil_moisture_old, soil_moisture_critical, wilting_point, root_fract_layer1, root_fract_layer2)


            # Update W_i:
            #W_i += (current_precipitation[np.newaxis, :, :] - E_c[:, :, :]) # Water stored in canopy, evap happens on previous W_i, so before precipitation in current timestep
            #W_i = np.minimum(W_i, W_im) # Ensure W_i doesn’t go above W_im
            #W_i = np.maximum(0, W_i)    # Ensure W_i doesn’t go below 0

            #throughfall = np.maximum(0, W_i - W_im)


            # Write results to the NetCDF file from the GPU
            prec_scaled[:, :, day] = Array(prec_gpu)
            tair_scaled[:, :, day] = Array(tair_gpu)

        end

        day_prev = day
        month_prev = month

    end

    println("Close output file...")
    close(out_ds)

    println("============ Completed run for year: $year ============\n")

    println("Postprocessing for year $year:")
    compress_file_async(output_file, 1)    

end

println("Done!")
