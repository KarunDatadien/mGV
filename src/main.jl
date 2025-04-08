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
    (b_infilt_cpu,   b_infilt_gpu)   = read_and_allocate_parameter(b_infilt_var)   
end

gpu_load_static_inputs([rmin_cpu, rarc_cpu, elev_cpu, ksat_cpu, residmoist_cpu, init_moist_cpu, root_cpu,
                        Wcr_cpu, Wfc_cpu, Wpwp_cpu, depth_cpu, quartz_cpu,
                        bulk_dens_cpu, soil_dens_cpu, expt_cpu, b_infilt_cpu],
                       [rmin_gpu, rarc_gpu, elev_gpu, ksat_gpu, residmoist_gpu, init_moist_gpu, root_gpu,
                        Wcr_gpu, Wfc_gpu, Wpwp_gpu, depth_gpu, quartz_gpu,
                        bulk_dens_gpu, soil_dens_gpu, expt_gpu, b_infilt_gpu])

reshape_static_inputs!()

# === Initial States ===
global water_storage = CUDA.fill(Float32(0.1), size(coverage_gpu))  # Fill with 0.1 on GPU
global throughfall   = CUDA.zeros(Float32, size(coverage_gpu))      # Allocate zeros on GPU
global bulk_dens_min = CUDA.zeros(Float32, size(bulk_dens_gpu))      
global soil_dens_min = CUDA.zeros(Float32, size(bulk_dens_gpu))      
global porosity      = CUDA.zeros(Float32, size(bulk_dens_gpu))      
global soil_moisture_new = CUDA.zeros(Float32, size(soil_dens_gpu))      
global soil_moisture_max = CUDA.zeros(Float32, size(soil_dens_gpu))    

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
    @time out_ds, prec_scaled, water_storage_output, Q12_output, tair_output, tsurf_output, 
    canopy_evaporation_output, canopy_evaporation_summed_output, transpiration_output, aerodynamic_resistance_output, 
    potential_evaporation_output, potential_evaporation_summed_output, net_radiation_output, 
    net_radiation_summed_output, max_water_storage_output, max_water_storage_summed_output, soil_evaporation_output = create_output_netcdf(output_file, prec_cpu, LAI_cpu)

    println("Running...") 
    num_days = size(prec_cpu, 3)
    day_prev = 0
    month_prev = 0

    # For the first timestep we set tsurf = tair (see page 14,421 of Liang et al. (1994)):
    tsurf = tair_gpu  # TODO: make tsurf global such that it's carried over to next year
    
    # Start year long loop with daily timestep
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

            aerodynamic_resistance = compute_aerodynamic_resistance(
                z2, d0_gpu, z0_gpu, K, tsurf, tair_gpu, wind_gpu
            ) # Eq. (3). 🚧 TODO: check once solve_surface_temperature function has been checked

            net_radiation = calculate_net_radiation(
                swdown_gpu, lwdown_gpu, albedo_gpu, tsurf
            ) # [W/m^2], works ✅

            potential_evaporation = calculate_potential_evaporation(
                tair_gpu, vp_gpu, elev_gpu, net_radiation, 
                aerodynamic_resistance, rarc_gpu
            ) # Penmann-Monteith equation with canopy resistance set to 0 , output in [mm/day], works ✅

            max_water_storage = calculate_max_water_storage(LAI_gpu) # Eq. (2), works ✅

            canopy_evaporation = calculate_canopy_evaporation(
                water_storage, max_water_storage, potential_evaporation, aerodynamic_resistance, rarc_gpu, prec_gpu
            ) # Eq. (1), (9) and (10), works ✅

            soil_moisture_old = soil_moisture_new

            transpiration = calculate_transpiration(
                potential_evaporation, aerodynamic_resistance, rarc_gpu, 
                water_storage, max_water_storage, soil_moisture_old, soil_moisture_critical, wilting_point, 
                root_gpu
            ) # Eq. (5), 🚧 TODO: add the gsm_inv multiplication
            
            # === Update Water Storage with Throughfall Computation; Eq. 16 ===
            new_water_storage = water_storage .+ (prec_gpu .- canopy_evaporation)
            throughfall = max.(0, new_water_storage .- max_water_storage)
            water_storage = clamp.(new_water_storage, 0, max_water_storage)

            soil_evaporation = calculate_soil_evaporation(soil_moisture_old, soil_moisture_max, potential_evaporation, b_infilt_gpu)
            println("soil_evaporation size: ", size(soil_evaporation))
            println("soil_moisture_old size: ", size(soil_moisture_old))
            println("soil_moisture_new size: ", size(soil_moisture_new))
            println("b_infilt_gpu size: ", size(b_infilt_gpu))
            println("potential_evaporation size: ", size(potential_evaporation))
            println("potential_evaporation[:,:,:,end] size: ", size(potential_evaporation[:,:,:,end]))

            # soil_evaporation size: (204, 180, 3)
            # soil_moisture_old size: (204, 180, 3)
            # soil_moisture_new size: (204, 180, 3)
            # potential_evaporation size: (204, 180, 1, 22)
            # potential_evaporation[:,:,:,end] size: (204, 180, 1)

            ## Drainage from layer 1 to 2; Q12
            Q12 = ksat_gpu[:, :, 1] .* ((soil_moisture_old[:, :, 1:2] .- residmoist_gpu[:, :, 1]) ./ (soil_moisture_max[:, :, 1] .- residmoist_gpu[:, :, 1])) .^ expt_gpu[:, :, 1] # Eq. (20)
            println("Q12 size: ", size(Q12))


            ## Water balance layer 1 and direct surface runoff
            #Q_surface = gw.calculate_Q_surface(current_precipitation, time_step_placeholder, soil_moisture_max[0,:,:], soil_moisture_old[0,:,:], i_0, max_infil)
    
            #soil_moisture_new[0,:,:] = soil_moisture_old[0,:,:] + (current_precipitation - Q_surface[0,:,:] - Q12[0,:,:] - E1)*time_step_placeholder
            
            # temporary check:
            soil_moisture_new[:,:,1:2] = soil_moisture_old[:,:,1:2] .+ prec_gpu

            ## Water balance layer 2 and subsurface runoff
            #E2 = 0 # Liang et al.: "Evaporation from bare soil is extracted only from layer 1; bare soil evaporation from layer 2 (E2) is assumed to be zero."
            #Q_subsurface = gw.calculate_Q_subsurface(Ds, Dsmax, soil_moisture_old[1,:,:], Ws, soil_moisture_max[1,:,:], soil_moisture_old[1,:,:])
            #soil_moisture_new[1,:,:] = soil_moisture_old[1,:,:] + (Q12[1,:,:] - Q_subsurface - E2)*time_step_placeholder
            println("canopy_evaporation: ", size(canopy_evaporation))
            println("transpiration: ", size(transpiration))

            E_n = canopy_evaporation + transpiration # + soil_evaporation
    
            ice_frac = 0

            ## TODO: fix placeholders:
            soil_temp1_gpu = copy(tair_gpu)
            soil_temp2_gpu = copy(tair_gpu)
    
            kappa_array = soil_conductivity(soil_moisture_new, ice_frac, soil_dens_min, bulk_dens_min, quartz_gpu, organic_frac, porosity)
            println("kappa_array has NaN: ", any(isnan, kappa_array), " min/max: ", minimum(kappa_array), " / ", maximum(kappa_array))

            
            Cs_array = volumetric_heat_capacity(bulk_dens_gpu ./ soil_dens_gpu, soil_moisture_new, ice_frac, organic_frac)

    #        tsurf = solve_surface_temperature(
    #            tair_gpu,          # Air temperature (CuArray)
    #            soil_temp1_gpu,    # Soil layer 1 temperature (CuArray)
    #            soil_temp2_gpu,    # Soil layer 2 temperature (CuArray)
    #            albedo_gpu,        # Surface albedo (CuArray)
    #            swdown_gpu,   # Rs: shortwave radiation (Float32)
    #            lwdown_gpu,   # RL: longwave radiation (Float32)
    #            aerodynamic_resistance, # Aerodynamic resistance (CuArray)
    #            kappa_array,         # Thermal conductivity (CuArray)
    #            0.1,             # D1 (layer 1 depth in meters) TODO: get from input file?
    #            0.4,             # D2 (layer 2 depth in meters) TODO: get from input file?
    #            day_sec,               # delta_t (time step in seconds -> 1 day in s)
    #            Cs_array,
    #            E_n
    #        ) # TODO: issue: somehow changes canopy_evaporation and others from shape: (204, 180, 1, 22) to (204, 180, 3, 22)

            # Write results to the NetCDF file from the GPU
            #water_storage_summed_output[:, :, day] = Array(sum(water_storage; dims=4)[:, :, :, 1])
            println("tsurf size: ", size(tsurf))
            println("tair size: ", size(tair_gpu))

            potential_evaporation_output[:, :, day, :] = Array(potential_evaporation)

            # Sum while skipping NaN
            summed_on_gpu = dropdims(
                sum(x -> isnan(x) ? 0.0 : x, potential_evaporation, dims=4) .* 
                (sum(x -> isnan(x) ? 0 : 1, potential_evaporation, dims=4) .> 0), 
                dims=4)
            # Replace all-NaN sums with NaN
            summed_on_gpu = ifelse.(sum(x -> isnan(x) ? 0 : 1, potential_evaporation, dims=4) .== 0, NaN, summed_on_gpu) # TODO: pre-allocate this?

            # Transfer to CPU and assign
            potential_evaporation_summed_output[:, :, day] = Array(summed_on_gpu)

            println("net_radiation: ", size(net_radiation))

            # Replace extreme values with NaN to avoid overflow issues and allow proper summing
            net_radiation .= ifelse.(
                abs.(net_radiation) .> 1e25,  # or 1e30 if you want stricter
                NaN,
                net_radiation
            )
                        
            net_radiation_output[:, :, day, :] = Array(net_radiation)

            # Sum while skipping NaN
            summed_on_gpu2 = dropdims(
                sum(x -> isnan(x) ? 0.0 : x, net_radiation, dims=4) .* 
                (sum(x -> isnan(x) ? 0 : 1, net_radiation, dims=4) .> 0), 
                dims=4)
            # Replace all-NaN sums with NaN
            summed_on_gpu2 = ifelse.(sum(x -> isnan(x) ? 0 : 1, net_radiation, dims=4) .== 0, NaN, summed_on_gpu2) # TODO: pre-allocate this?

            net_radiation_summed_output[:, :, day] = Array(summed_on_gpu2)

            println("aerodynamic_resistance: ", size(aerodynamic_resistance))
            
            aerodynamic_resistance_output[:, :, day, :] = Array(aerodynamic_resistance)

            # Replace extreme values with NaN to avoid overflow issues and allow proper summing
            canopy_evaporation .= ifelse.(
                abs.(canopy_evaporation) .> 1e25,  # or 1e30 if you want stricter
                NaN,
                canopy_evaporation
            )

            canopy_evaporation_output[:, :, day, :] = Array(canopy_evaporation)

            # Sum while skipping NaN
            summed_on_gpu4 = dropdims(
                sum(x -> isnan(x) ? 0.0 : x, canopy_evaporation, dims=4) .* 
                (sum(x -> isnan(x) ? 0 : 1, canopy_evaporation, dims=4) .> 0), 
                dims=4)
            # Replace all-NaN sums with NaN
            summed_on_gpu4 = ifelse.(sum(x -> isnan(x) ? 0 : 1, canopy_evaporation, dims=4) .== 0, NaN, summed_on_gpu4) # TODO: pre-allocate this?

            canopy_evaporation_summed_output[:, :, day] = Array(summed_on_gpu4)


            transpiration_output[:, :, day, :] = Array(transpiration)

            tair_output[:, :, day] = Array(tair_gpu)
          #  tsurf_output[:, :, day, :, :] = Array(tsurf)
            water_storage_output[:, :, day, :] = Array(water_storage)
            prec_scaled[:, :, day] = Array(prec_gpu)
            #Q12_output[:, :, day] = Array(Q12)



            # Replace extreme values with NaN to avoid overflow issues and allow proper summing
            max_water_storage .= ifelse.(
                abs.(max_water_storage) .> 1e25,  # or 1e30 if you want stricter
                NaN,
                max_water_storage
            )

            max_water_storage_output[:, :, day, :] = Array(max_water_storage)

            # Sum while skipping NaN
            summed_on_gpu3 = dropdims(
                sum(x -> isnan(x) ? 0.0 : x, max_water_storage, dims=4) .* 
                (sum(x -> isnan(x) ? 0 : 1, max_water_storage, dims=4) .> 0), 
                dims=4)
            # Replace all-NaN sums with NaN
            summed_on_gpu3 = ifelse.(sum(x -> isnan(x) ? 0 : 1, max_water_storage, dims=4) .== 0, NaN, summed_on_gpu3) # TODO: pre-allocate this?
            
            max_water_storage_summed_output[:, :, day] = Array(summed_on_gpu3)

            println("soil_evaporation: ", size(soil_evaporation))
            soil_evaporation_output[: , :, day, :] = Array(soil_evaporation)

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