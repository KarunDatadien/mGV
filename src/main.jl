using .SimConstants
const to = TimerOutputs.TimerOutput()

println("Loading parameter data and allocating memory...")

@time begin
    (d0_cpu,         d0_gpu)         = read_and_allocate_parameter(d0_var)
    (z0_cpu,         z0_gpu)         = read_and_allocate_parameter(z0_var)
    (LAI_cpu,        LAI_gpu)        = read_and_allocate_parameter(LAI_var)
    (albedo_cpu,     albedo_gpu)     = read_and_allocate_parameter(albedo_var) 
    (rmin_cpu,       rmin_gpu)       = read_and_allocate_parameter(rmin_var) 
    (rarc_cpu,       rarc_gpu)       = read_and_allocate_parameter(rarc_var) 
    (elev_cpu,       elev_gpu)       = read_and_allocate_parameter(elev_var) 
    (ksat_cpu,       ksat_gpu)       = read_and_allocate_parameter(ksat_var) 
    (residmoist_cpu, residmoist_gpu) = read_and_allocate_parameter(residmoist_var) 
    (init_moist_cpu, init_moist_gpu) = read_and_allocate_parameter(init_moist_var) 
    (root_cpu,       root_gpu)       = read_and_allocate_parameter(root_var) 
    (Wcr_cpu,        Wcr_gpu)        = read_and_allocate_parameter(Wcr_var)
    (Wfc_cpu,        Wfc_gpu)        = read_and_allocate_parameter(Wfc_var)
    (Wpwp_cpu,       Wpwp_gpu)       = read_and_allocate_parameter(Wpwp_var)
    (depth_cpu,      depth_gpu)      = read_and_allocate_parameter(depth_var)
    (quartz_cpu,     quartz_gpu)     = read_and_allocate_parameter(quartz_var)
    (bulk_dens_cpu,  bulk_dens_gpu)  = read_and_allocate_parameter(bulk_dens_var)
    (soil_dens_cpu,  soil_dens_gpu)  = read_and_allocate_parameter(soil_dens_var)
    (expt_cpu,       expt_gpu)       = read_and_allocate_parameter(expt_var)
    (coverage_cpu,   coverage_gpu)   = read_and_allocate_parameter(coverage_var)
    (b_infilt_cpu,   b_infilt_gpu)   = read_and_allocate_parameter(b_infilt_var)   
    (Ds_cpu,         Ds_gpu)         = read_and_allocate_parameter(Ds_var)   
    (Dsmax_cpu,      Dsmax_gpu)      = read_and_allocate_parameter(Dsmax_var)   
    (Ws_cpu,         Ws_gpu)         = read_and_allocate_parameter(Ws_var)   
    (dp_cpu,         dp_gpu)         = read_and_allocate_parameter(dp_var)
    (Tavg_cpu,       Tavg_gpu)       = read_and_allocate_parameter(Tavg_var)
end

println("Loading gpu_load_static_inputs...")
gpu_load_static_inputs([rmin_cpu, rarc_cpu, elev_cpu, ksat_cpu, residmoist_cpu, init_moist_cpu, root_cpu,
                        Wcr_cpu, Wfc_cpu, Wpwp_cpu, depth_cpu, quartz_cpu,
                        bulk_dens_cpu, soil_dens_cpu, expt_cpu, b_infilt_cpu,
                        Ds_cpu, Dsmax_cpu, Ws_cpu, dp_cpu, Tavg_cpu],
                       [rmin_gpu, rarc_gpu, elev_gpu, ksat_gpu, residmoist_gpu, init_moist_gpu, root_gpu,
                        Wcr_gpu, Wfc_gpu, Wpwp_gpu, depth_gpu, quartz_gpu,
                        bulk_dens_gpu, soil_dens_gpu, expt_gpu, b_infilt_gpu,
                        Ds_gpu, Dsmax_gpu, Ws_gpu, dp_gpu, Tavg_gpu])

reshape_static_inputs!()

# === Initial States ===
global water_storage = CUDA.fill(Float32(0.1), size(coverage_gpu))  # Fill with 0.1 on GPU
global throughfall   = CUDA.zeros(Float32, size(coverage_gpu))      # Allocate zeros on GPU
global bulk_dens_min = CUDA.zeros(Float32, size(bulk_dens_gpu))
global soil_dens_min = CUDA.zeros(Float32, size(bulk_dens_gpu))
global porosity      = CUDA.zeros(Float32, size(bulk_dens_gpu))
global soil_moisture_old = CUDA.zeros(Float32, size(soil_dens_gpu))
global soil_moisture_new = CUDA.zeros(Float32, size(soil_dens_gpu))
global soil_moisture_max = CUDA.zeros(Float32, size(soil_dens_gpu))
global soil_temperature = CUDA.zeros(Float32, size(soil_dens_gpu))
global Lsum = CUDA.zeros(Float32, size(soil_dens_gpu))
global tsurf = CUDA.zeros(Float64, size(d0_gpu))
global tsurf_n = CUDA.zeros(Float64, size(albedo_gpu))

soil_moisture_new = init_moist_gpu
soil_temperature[:, :, 1:1] .= Tavg_gpu
soil_temperature[:, :, 2:2] .= Tavg_gpu
soil_temperature[:, :, 3:3] .= Tavg_gpu

# === Calculate Soil Properties ===
bulk_dens_min, soil_dens_min, porosity, soil_moisture_max, soil_moisture_critical, field_capacity, wilting_point = 
    calculate_soil_properties(bulk_dens_gpu, soil_dens_gpu, depth_gpu, Wcr_gpu, Wfc_gpu, Wpwp_gpu)

function process_year(year)
    
    # Ensure we're modifying the global variables
    global water_storage, throughfall, bulk_dens_min, soil_dens_min, porosity, soil_moisture_old, 
           soil_moisture_new, soil_moisture_max, soil_temperature, Lsum, tsurf, tsurf_n

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
    
    @time out_ds, precipitation_output, water_storage_output, Q12_output,
          tair_output, tsurf_output, tsurf_summed_output, canopy_evaporation_output,
          canopy_evaporation_summed_output, transpiration_output, aerodynamic_resistance_output,
          potential_evaporation_output, potential_evaporation_summed_output, net_radiation_output,
          net_radiation_summed_output, max_water_storage_output, max_water_storage_summed_output,
          soil_evaporation_output, soil_temperature_output, soil_moisture_output =
          create_output_netcdf(output_file, prec_cpu, LAI_cpu)

    println("Running...") 
    num_days = size(prec_cpu, 3)
    day_prev = 0
    month_prev = 0

    # Start year long loop with daily timestep
    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        @timeit to "process_year" begin
        month = day_to_month(day, year)
        
        if GPU_USE == true
            # Explicitly copy preloaded data to the GPU
            @timeit to "gpu_load_monthly_inputs" gpu_load_monthly_inputs(month, month_prev, 
                                    [d0_cpu, z0_cpu, LAI_cpu, albedo_cpu, coverage_cpu], 
                                    [d0_gpu, z0_gpu, LAI_gpu, albedo_gpu, coverage_gpu])
            @timeit to "gpu_load_daily_inputs" gpu_load_daily_inputs(day, day_prev, 
                                  [prec_cpu, tair_cpu, wind_cpu, vp_cpu, swdown_cpu, lwdown_cpu], 
                                  [prec_gpu, tair_gpu, wind_gpu, vp_gpu, swdown_gpu, lwdown_gpu])
                                  
            # For the first timestep we set tsurf = tair (see page 14,421 of Liang et al. (1994)):
            if day == 1 && year == start_year 
                tsurf = tair_gpu  # Set tsurf = tair_gpu on first timestep
                tsurf_n = repeat(tsurf, outer=(1, 1, 1, nveg))  # (204, 180, 1, 22)
            end

            @timeit to "compute_aerodynamic_resistance" aerodynamic_resistance = compute_aerodynamic_resistance(
                z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu
            ) # Eq. (3). 🚧 TODO: check once solve_surface_temperature function has been checked

            @timeit to "calculate_net_radiation" net_radiation = calculate_net_radiation(
                swdown_gpu, lwdown_gpu, albedo_gpu, tsurf
            ) # [W/m^2], works ✅

            @timeit to "calculate_potential_evaporation" potential_evaporation = calculate_potential_evaporation(
                tair_gpu, vp_gpu, elev_gpu, net_radiation, 
                aerodynamic_resistance, rarc_gpu
            ) # Penmann-Monteith equation with canopy resistance set to 0 , output in [mm/day], works ✅

            @timeit to "calculate_max_water_storage" max_water_storage = calculate_max_water_storage(LAI_gpu) # Eq. (2), works ✅ TODO: this only needs to be calculated ONCE, so put outside the loop

            @timeit to "calculate_canopy_evaporation" canopy_evaporation = calculate_canopy_evaporation(
                water_storage, max_water_storage, potential_evaporation, aerodynamic_resistance, rarc_gpu, prec_gpu
            ) # Eq. (1), (9) and (10), works ✅

            soil_moisture_old = soil_moisture_new

            @timeit to "calculate_transpiration" transpiration = calculate_transpiration(
                potential_evaporation, aerodynamic_resistance, rarc_gpu, 
                water_storage, max_water_storage, soil_moisture_old, soil_moisture_critical, wilting_point, 
                root_gpu
            ) # Eq. (5), 🚧 TODO: add the gsm_inv multiplication
            
            @timeit to "calculate_soil_evaporation" soil_evaporation = calculate_soil_evaporation(soil_moisture_old, soil_moisture_max, potential_evaporation, b_infilt_gpu)
            
            # === Update Water Storage with Throughfall Computation; Eq. 16 ===
            @timeit to "update_water_canopy_storage" begin 
            new_water_storage = water_storage .+ (prec_gpu .- canopy_evaporation) 
            throughfall = max.(0, new_water_storage .- max_water_storage)
            water_storage = max.(0.0, min.(new_water_storage, max_water_storage)) # works ✅
            end

            # Direct surface runoff
            @timeit to "calculate_surface_runoff" surface_runoff = calculate_surface_runoff(prec_gpu, soil_moisture_old, soil_moisture_max, b_infilt_gpu) # Eq. (18a) and (18b)

            # Drainage from layer 1 to 2; Q12
            @timeit to "calculate_drainage_Q12" Q_12 = calculate_drainage_Q12(soil_moisture_old, soil_moisture_max, ksat_gpu, residmoist_gpu, porosity) # Eq. (20) TODO: check if porosity is the correct argument here?

            # Water balance toplayer update 
            @timeit to "update_topsoil_moisture" soil_moisture_new = update_topsoil_moisture(prec_gpu, soil_moisture_old, soil_moisture_max, surface_runoff, Q_12, soil_evaporation, depth_gpu)

            # Calculate subsurface runoff
            @timeit to "calculate_subsurface_runoff" subsurface_runoff = calculate_subsurface_runoff(soil_moisture_old, soil_moisture_max, Ds_gpu, Dsmax_gpu, Ws_gpu) # Eq. (21)           

            # Update bottom soil moisture and get total subsurface runoff
            @timeit to "update_bottomsoil_moisture" soil_moisture_new[:, :, 3:3], subsurface_runoff_total = update_bottomsoil_moisture(
                soil_moisture_old, soil_moisture_max, subsurface_runoff, Q_12
            ) # Eq. (22), TODO: compare this with the paper

            # TODO: implement section 2.5
            
            @timeit to "E_n sum"                 E_n              = canopy_evaporation .+ transpiration  # + soil_evaporation
            @timeit to "Set ice_frac sum"        ice_frac         = 0.0
            @timeit to "soil_conductivity"       kappa_array      = soil_conductivity(
                                                                   soil_moisture_new, ice_frac, soil_dens_min,
                                                                   bulk_dens_min, quartz_gpu, organic_frac, porosity)
            @timeit to "volumetric_heat_capacity" Cs_array        = volumetric_heat_capacity(
                                                                   bulk_dens_gpu ./ soil_dens_gpu,
                                                                   soil_moisture_new, ice_frac, organic_frac)
            @timeit to "estimate_layer_temperature" soil_temperature = estimate_layer_temperature(
                                                                   depth_gpu, dp_gpu, tsurf, soil_temperature, Tavg_gpu)
            

            #@timeit to "solve_surface_temperature" tsurf_n = solve_surface_temperature(
            #    tsurf_n,          # Surface temperature (CuArray)
            #    soil_temperature[:,:,1],    # Soil layer 1 temperature (CuArray) TODO: should be sum of 1 and 2
            #    soil_temperature[:,:,2],    # Soil layer 2 temperature (CuArray) TODO: should be layer 3 (bottom)
            #    albedo_gpu,        # Surface albedo (CuArray)
            #    swdown_gpu,   # Rs: shortwave radiation (Float32)
            #    lwdown_gpu,   # RL: longwave radiation (Float32)
            #    aerodynamic_resistance, # Aerodynamic resistance (CuArray)
            #    kappa_array,         # Thermal conductivity (CuArray)
            #    depth_gpu[:,:,1],             # D1 (layer 1 depth in meters) TODO: should be sum of 1 and 2
            #    depth_gpu[:,:,2],             # D2 (layer 2 depth in meters) TODO: should be layer 3 (bottom)
            #    day_sec,               # delta_t (time step in seconds -> 1 day in s)
            #    Cs_array,
            #    E_n
            #) # TODO: issue: somehow changes canopy_evaporation and others from shape: (204, 180, 1, 22) to (204, 180, 3, 22)

            # Convert tsurf_n to Float64
            tsurf_n = convert(CuArray{Float64, 4}, tsurf_n) # TODO: do this in a more sensible way, somewhere else

            @timeit to "outputs" begin
                ### Variables without fill-value replacement ###
                @timeit to "tsurf_output"                      tsurf_output[:, :, day, :]                = Array(tsurf_n)
                @timeit to "tsurf_summed_output"               tsurf_summed_output[:, :, day]            = Array(sum_with_nan_handling(tsurf_n, 4))
                @timeit to "potential_evaporation_output"      potential_evaporation_output[:, :, day, :] = Array(potential_evaporation)
                @timeit to "potential_evaporation_summed"      potential_evaporation_summed_output[:, :, day] = Array(sum_with_nan_handling(potential_evaporation, 4))
                @timeit to "aerodynamic_resistance_output"     aerodynamic_resistance_output[:, :, day, :] = Array(aerodynamic_resistance)
                @timeit to "transpiration_output"              transpiration_output[:, :, day, :]        = Array(transpiration)
                @timeit to "tair_output"                       tair_output[:, :, day]                    = Array(tair_gpu)
                @timeit to "water_storage_output"              water_storage_output[:, :, day, :]        = Array(water_storage)
                @timeit to "precipitation_output"              precipitation_output[:, :, day]           = Array(prec_gpu)
                @timeit to "Q12_output"                        Q12_output[:, :, day]                     = Array(Q_12)
                @timeit to "soil_evaporation_output"           soil_evaporation_output[:, :, day, :]     = Array(soil_evaporation)
                @timeit to "soil_temperature_output"           soil_temperature_output[:, :, day, :]     = Array(soil_temperature)
                @timeit to "soil_moisture_output"              soil_moisture_output[:, :, day, :]        = Array(soil_moisture_old)
            
                ### Variables with fill-value replacement ###
                @timeit to "net_radiation_processed"           net_radiation_processed = ifelse.(abs.(net_radiation) .> 1e25, NaN, net_radiation)
                @timeit to "net_radiation_output"              net_radiation_output[:, :, day, :]        = Array(net_radiation_processed)
                @timeit to "net_radiation_summed_output"       net_radiation_summed_output[:, :, day]    = Array(sum_with_nan_handling(net_radiation_processed, 4))
            
                @timeit to "canopy_evaporation_processed"      canopy_evaporation_processed = ifelse.(abs.(canopy_evaporation) .> 1e25, NaN, canopy_evaporation)
                @timeit to "canopy_evaporation_output"         canopy_evaporation_output[:, :, day, :]   = Array(canopy_evaporation_processed)
                @timeit to "canopy_evaporation_summed_output"  canopy_evaporation_summed_output[:, :, day] = Array(sum_with_nan_handling(canopy_evaporation_processed, 4))
            
                @timeit to "max_water_storage_processed"       max_water_storage_processed = ifelse.(abs.(max_water_storage) .> 1e25, NaN, max_water_storage)
                @timeit to "max_water_storage_output"          max_water_storage_output[:, :, day, :]    = Array(max_water_storage_processed)
                @timeit to "max_water_storage_summed_output"   max_water_storage_summed_output[:, :, day] = Array(sum_with_nan_handling(max_water_storage_processed, 4))
            end
            
        end # gpu use
        
        day_prev = day
        month_prev = month
    end # timeit year loop

    end # year loop

    println("Closing output file...")
    @timeit to "closing outputfile" close(out_ds)
    
    println("============ Completed run for year: $year ============\n")
    println("Postprocessing for year $year...")
    @timeit to "compress_file_async call" compress_file_async(output_file, 1)
end

# Main loop over years:
for year in start_year:end_year
    if has_input_files(year)
        process_year(year)
        
        show(to) # Print profiling data

        @timeit to "garbage collection" begin 
            Base.GC.gc()  # Force garbage collection after processing each year TODO: does this actually do anything?
            CUDA.reclaim() # TODO: check does this actually do anything?
        end
    else
        println("Skipping year $year due to missing input files.")
    end
end
println("Done!")