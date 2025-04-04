using .SimConstants
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
end

# === Initial States ===
global water_storage = CUDA.fill(Float32(0.1), size(coverage_gpu))  # Fill with 0.1 on GPU
global throughfall   = CUDA.zeros(Float32, size(coverage_gpu))      # Allocate zeros on GPU
global bulk_dens_min = CUDA.zeros(Float32, size(bulk_dens_gpu))      
global soil_dens_min = CUDA.zeros(Float32, size(bulk_dens_gpu))      
global porosity      = CUDA.zeros(Float32, size(bulk_dens_gpu))      
global soil_moisture_new = CUDA.zeros(Float32, size(soil_dens_gpu))      
global soil_moisture_max = CUDA.zeros(Float32, size(soil_dens_gpu))      


gpu_load_static_inputs([rmin_cpu, rarc_cpu, elev_cpu, ksat_cpu, residmoist_cpu, init_moist_cpu, root_cpu,
                        Wcr_cpu, Wfc_cpu, Wpwp_cpu, depth_cpu, quartz_cpu,
                        bulk_dens_cpu, soil_dens_cpu, expt_cpu],
                       [rmin_gpu, rarc_gpu, elev_gpu, ksat_gpu, residmoist_gpu, init_moist_gpu, root_gpu,
                        Wcr_gpu, Wfc_gpu, Wpwp_gpu, depth_gpu, quartz_gpu,
                        bulk_dens_gpu, soil_dens_gpu, expt_gpu])

reshape_static_inputs!()

soil_moisture_new = init_moist_gpu

# === Calculate Soil Properties ===
bulk_dens_min, soil_dens_min, porosity, soil_moisture_max, soil_moisture_critical, field_capacity, wilting_point = 
    calculate_soil_properties(bulk_dens_gpu, soil_dens_gpu, depth_gpu, Wcr_gpu, Wfc_gpu, Wpwp_gpu)

function process_year(year)
    global water_storage  # Ensure we're modifying the global variables
    global throughfall  

    global bulk_dens_min
    global soil_dens_min
    global porosity
    global soil_moisture_new
    global soil_moisture_max

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
    @time out_ds, prec_scaled, water_storage_output, Q12_output, tair_output, tsurf_output, canopy_evaporation_output, transpiration_output, aerodynamic_resistance_output = create_output_netcdf(output_file, prec_cpu, LAI_cpu)

    println("Running...") 
    num_days = size(prec_cpu, 3)
    day_prev = 0
    month_prev = 0

    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        month = day_to_month(day, year)
        
        if GPU_USE == true
            # Explicitly copy preloaded data to the GPU
            gpu_load_monthly_inputs(month, month_prev, 
                                    [d0_cpu, z0_cpu, LAI_cpu, albedo_cpu, coverage_cpu], 
                                    [d0_gpu, z0_gpu, LAI_gpu, albedo_gpu, coverage_gpu])
            gpu_load_daily_inputs(day, day_prev, 
                                  [prec_cpu, tair_cpu, wind_cpu, vp_cpu, swdown_cpu, lwdown_cpu], 
                                  [prec_gpu, tair_gpu, wind_gpu, vp_gpu, swdown_gpu, lwdown_gpu])

            # Start computations for potential evaporation
            tsurf = tair_gpu  # TODO: calculate actual tsurf

            aerodynamic_resistance = compute_aerodynamic_resistance(
                z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu
            ) # Eq. (3) to check

            canopy_resistance = compute_canopy_resistance(rmin_gpu, LAI_gpu) # Eq. (6), to check

            net_radiation = calculate_net_radiation(
                swdown_gpu, lwdown_gpu, albedo_gpu, tsurf
            )

            #println("aerodynamic_resistance SIZE:", size(aerodynamic_resistance)) 
            #println("canopy_resistance SIZE:", size(canopy_resistance))
            #println("rarc_gpu SIZE:", size(rarc_gpu)) 

            potential_evaporation = calculate_potential_evaporation(
                tair_gpu, vp_gpu, elev_gpu, net_radiation, 
                aerodynamic_resistance, canopy_resistance, rarc_gpu
            ) # Penmann-Monteith equation, to check more precisely

            max_water_storage = calculate_max_water_storage(LAI_gpu) # Eq. (2), checked

            canopy_evaporation = calculate_canopy_evaporation(
                water_storage, max_water_storage, LAI_gpu, potential_evaporation, aerodynamic_resistance, rarc_gpu, prec_gpu
            ) #Eq. (1), (9) and (10), to check: Eq. (10)

            soil_moisture_old = soil_moisture_new

            transpiration = calculate_transpiration(
                potential_evaporation, aerodynamic_resistance, rarc_gpu, canopy_resistance, 
                water_storage, max_water_storage, soil_moisture_old, soil_moisture_critical, wilting_point, 
                root_gpu
            )
            
            # === Update Water Storage ===
            throughfall .= 0
            water_storage .+= (prec_gpu .- canopy_evaporation)
            water_storage .= clamp.(water_storage, 0, max_water_storage)
            throughfall = max.(0, water_storage .- max_water_storage)             # Compute throughfall
        
            ## Drainage from layer 1 to 2; Q12
            Q12 = ksat_gpu[:, :, 1] .* ((soil_moisture_old[:, :, 1] .- residmoist_gpu[:, :, 1]) ./ (soil_moisture_max[:, :, 1] .- residmoist_gpu[:, :, 1])) .^ expt_gpu[:, :, 1]

            ## Calculate intermediate values
            #precipitation_term = current_precipitation * time_step_placeholder
    
            ## Water balance layer 1 and direct surface runoff
            #Q_surface = gw.calculate_Q_surface(current_precipitation, time_step_placeholder, soil_moisture_max[0,:,:], soil_moisture_old[0,:,:], i_0, max_infil)
    
            #soil_moisture_new[0,:,:] = soil_moisture_old[0,:,:] + (current_precipitation - Q_surface[0,:,:] - Q12[0,:,:] - E1)*time_step_placeholder
            
            # temporary check:
            soil_moisture_new[:,:,1] = soil_moisture_old[:,:,1] .+ 1.0 #.+ prec_gpu

            ## Water balance layer 2 and subsurface runoff
            #E2 = 0 # Liang et al.: "Evaporation from bare soil is extracted only from layer 1; bare soil evaporation from layer 2 (E2) is assumed to be zero."
            #Q_subsurface = gw.calculate_Q_subsurface(Ds, Dsmax, soil_moisture_old[1,:,:], Ws, soil_moisture_max[1,:,:], soil_moisture_old[1,:,:])
            #soil_moisture_new[1,:,:] = soil_moisture_old[1,:,:] + (Q12[1,:,:] - Q_subsurface - E2)*time_step_placeholder
            println("canopy_evaporation: ", size(canopy_evaporation))
            println("transpiration: ", size(transpiration))

            E_n = canopy_evaporation + transpiration
    
            ice_frac = 0

            ## Placeholders:
            soil_temp1_gpu = copy(tair_gpu)
            soil_temp2_gpu = copy(tair_gpu)            
    
            kappa_array = soil_conductivity(soil_moisture_new, ice_frac, soil_dens_min, bulk_dens_min, quartz_gpu, soil_dens_gpu, bulk_dens_gpu, organic_frac, porosity)
            println("kappa_array has NaN: ", any(isnan, kappa_array), " min/max: ", minimum(kappa_array), " / ", maximum(kappa_array))

            
            Cs_array = volumetric_heat_capacity(bulk_dens_gpu ./ soil_dens_gpu, soil_moisture_new, ice_frac, organic_frac)

            tsurf = solve_surface_temperature(
                tair_gpu,          # Air temperature (CuArray)
                soil_temp1_gpu,    # Soil layer 1 temperature (CuArray)
                soil_temp2_gpu,    # Soil layer 2 temperature (CuArray)
                albedo_gpu,        # Surface albedo (CuArray)
                swdown_gpu,   # Rs: shortwave radiation (Float32)
                lwdown_gpu,   # RL: longwave radiation (Float32)
                aerodynamic_resistance, # Aerodynamic resistance (CuArray)
                kappa_array,         # Thermal conductivity (CuArray)
                0.1,             # D1 (layer 1 depth in meters)
                0.4,             # D2 (layer 2 depth in meters)
                day_sec,               # delta_t (time step in seconds -> 1 day in s)
                elev_gpu,          # Elevation (CuArray)
                vp_gpu,            # Vapor pressure (CuArray)
                Cs_array,
                E_n
            )

            # Write results to the NetCDF file from the GPU
            #water_storage_summed_output[:, :, day] = Array(sum(water_storage; dims=4)[:, :, :, 1])
            println("tsurf size: ", size(tsurf))
            println("tair size: ", size(tair_gpu))

            aerodynamic_resistance_output[:, :, day, :] = Array(aerodynamic_resistance)
            canopy_evaporation_output[:, :, day, :] = Array(canopy_evaporation)
            transpiration_output[:, :, day, :] = Array(transpiration)
            tair_output[:, :, day] = Array(tair_gpu)
            tsurf_output[:, :, day, :, :] = Array(tsurf)
            water_storage_output[:, :, day, :] = Array(water_storage)
            prec_scaled[:, :, day] = Array(prec_gpu)
            Q12_output[:, :, day] = Array(Q12)

        end
        
        day_prev = day
        month_prev = month
    end

    println("Closing output file...")
    close(out_ds)
    
    println("============ Completed run for year: $year ============\n")
    println("Postprocessing for year $year...")
    compress_file_async(output_file, 1)
end

# Main loop over years:
for year in start_year:end_year
    if has_input_files(year)
        process_year(year)
        GC.gc()  # Force garbage collection after processing each year
    else
        println("Skipping year $year due to missing input files.")
    end
end

println("Done!")