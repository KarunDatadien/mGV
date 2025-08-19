using .SimConstants
const to = TimerOutputs.TimerOutput()

println("Loading parameter data and allocating memory...")

@time begin
    @load_params(
        lat, lon, d0, z0, z0soil, LAI, albedo, rmin, rarc, cv, elev,
        ksat, residmoist, init_moist, root, Wcr, Wfc, Wpwp, depth,
        quartz, bulk_dens, soil_dens, expt, coverage, b_infilt,
        Ds, Dsmax, Ws, dp, Tavg, c_expt
    )
end

@timeit to "gpu_load_static_inputs" gpu_load_static_inputs(@vars(
    rmin, rarc, cv, elev, ksat, residmoist, init_moist, root, Wcr, Wfc, Wpwp,
    depth, quartz, bulk_dens, soil_dens, expt, b_infilt, Ds, Dsmax, Ws, dp, Tavg, z0soil
)...)

reshape_static_inputs!()

# === Initial States ===
global water_storage = CUDA.zeros(float_type, size(coverage_gpu))  # Fill with 0.0 on GPU
global throughfall   = CUDA.zeros(float_type, size(Ds_gpu))      # Allocate zeros on GPU
global canopy_evaporation = CUDA.zeros(float_type, size(coverage_gpu))
global bulk_dens_min = CUDA.zeros(float_type, size(bulk_dens_gpu))
global soil_dens_min = CUDA.zeros(float_type, size(bulk_dens_gpu))
global porosity      = CUDA.zeros(float_type, size(bulk_dens_gpu))
global soil_temperature = CUDA.zeros(float_type, size(soil_dens_gpu))
global Lsum = CUDA.zeros(float_type, size(soil_dens_gpu))
global tsurf = CUDA.zeros(float_type, size(d0_gpu))
global Q_12 = CUDA.zeros(float_type, size(Tavg_gpu))

global soil_moisture_old = CUDA.zeros(float_type, size(soil_dens_gpu, 1), size(soil_dens_gpu, 2), size(soil_dens_gpu, 3))
global soil_moisture_new = CUDA.zeros(float_type, size(soil_dens_gpu, 1), size(soil_dens_gpu, 2), size(soil_dens_gpu, 3))
global soil_moisture_max = CUDA.zeros(float_type, size(soil_dens_gpu, 1), size(soil_dens_gpu, 2), size(soil_dens_gpu, 3))

soil_temperature[:, :, 1:1] = Tavg_gpu
soil_temperature[:, :, 2:2] = Tavg_gpu
soil_temperature[:, :, 3:3] = Tavg_gpu

# === Calculate Soil Properties ===
bulk_dens_min, soil_dens_min, porosity, soil_moisture_max, soil_moisture_critical, field_capacity, wilting_point, residual_moisture = 
    calculate_soil_properties(bulk_dens_gpu, soil_dens_gpu, depth_gpu, Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu)

# Repeat init_moist_gpu along the 4th dimension (TODO: note, I may change init_moist_gpu here to field_capacity after discussions with modelling group)
soil_moisture_new = init_moist_gpu
soil_moisture_old = init_moist_gpu
#soil_moisture_new = field_capacity, outer=(1, 1, 1, size(coverage_gpu, 4)))
soil_moisture_max = soil_moisture_max

# After you load static fields
println("Wmax (mean) [mm]: L1=", mean(Array(soil_moisture_max[:,:,1])),
                             " L2=", mean(Array(soil_moisture_max[:,:,2])),
                             " L3=", mean(Array(soil_moisture_max[:,:,3])))
println("Ksat (median) [mm/day]: L1=", median(Array(ksat_gpu[:,:,1])),
                                   " L2=", median(Array(ksat_gpu[:,:,2])),
                                   " L3=", median(Array(ksat_gpu[:,:,3])))

function process_year(year)

    # Ensure we're modifying the global variables
    global water_storage, throughfall, canopy_evaporation, bulk_dens_min, soil_dens_min, porosity, soil_moisture_old, Q_12,
           soil_moisture_new, soil_moisture_max, soil_moisture_critical, field_capacity, wilting_point, residual_moisture, soil_temperature, Lsum, tsurf

    println("============ Start run for year: $year ============")
    println("Loading forcing data and allocating memory...")
    @time begin
        @load_forcing year prec tair wind vp swdown lwdown
    end

    println("Opening output file...")
    output_file = joinpath(output_dir, "$(output_file_prefix)$(year).nc")
    
    @time out_ds, precipitation_output, water_storage_output, water_storage_summed_output, Q12_output, 
          tair_output, tsurf_output, canopy_evaporation_output, canopy_evaporation_summed_output, 
          transpiration_output, transpiration_summed_output, aerodynamic_resistance_output, aerodynamic_resistance_summed_output,
          potential_evaporation_output, potential_evaporation_summed_output, net_radiation_output,
          net_radiation_summed_output, max_water_storage_output, max_water_storage_summed_output,
          soil_evaporation_output, soil_temperature_output, soil_moisture_output, total_et_output, total_runoff_output, 
          kappa_array_output, cs_array_output, wilting_point_output, soil_moisture_max_output, soil_moisture_critical_output,
          E_1_t_output, E_2_t_output, g_sw_1_output, g_sw_2_output, g_sw_output, residual_moisture_output, 
          throughfall_output, throughfall_summed_output, topsoil_moisture_addition_output =
          create_output_netcdf(output_file, prec_cpu, LAI_cpu, float_type, lat_cpu, lon_cpu)

println("Soil moisture at position [3,21,1]: ", Array(soil_moisture_new[4:4, 22:22, 1:1])[1])
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
            @timeit to "gpu_load_monthly_inputs" gpu_load_monthly_inputs(month, month_prev, @vars(d0, z0, LAI, albedo, coverage)...)
            @timeit to "gpu_load_daily_inputs" gpu_load_daily_inputs(day, day_prev, @vars(prec, tair, wind, vp, swdown, lwdown)...)

            # For the first timestep we set tsurf = tair (see page 14,421 of Liang et al. (1994)):
            if day == 1 && year == start_year 
                tsurf = tair_gpu 
            end

            @timeit to "compute_aerodynamic_resistance" aerodynamic_resistance = compute_aerodynamic_resistance(
                z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu
            ) # Eq. (3). 🚧 TODO: check once solve_surface_temperature function has been checked

            @timeit to "calculate_net_radiation" net_radiation = calculate_net_radiation(
                swdown_gpu, lwdown_gpu, albedo_gpu, tsurf
            ) # [W/m^2], works ✅

            @timeit to "calculate_potential_evaporation" potential_evaporation = calculate_potential_evaporation(
                tair_gpu, vp_gpu, elev_gpu, net_radiation, 
                aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
            ) # Penmann-Monteith equation with canopy resistance set to 0, output in [mm/day], works ✅

            
            @timeit to "calculate_max_water_storage" max_water_storage = calculate_max_water_storage(LAI_gpu, cv_gpu, coverage_gpu) # Eq. (2), works ✅ TODO: this only needs to be calculated ONCE, so put outside the loop

            @timeit to "calculate_canopy_evaporation" canopy_evaporation, f_n = calculate_canopy_evaporation(
    water_storage, max_water_storage, potential_evaporation, aerodynamic_resistance, rarc_gpu, prec_gpu, cv_gpu, rmin_gpu, LAI_gpu, tair_gpu, elev_gpu
) # Eq. (1), (9) and (10), works ✅

            
            # TODO: transpiration, E_1_t and E_2_t output gives missing values on parts of the grid
@timeit to "calculate_transpiration" transpiration, E_1_t, E_2_t, g_sw_1, g_sw_2, g_sw = calculate_transpiration(
    potential_evaporation, aerodynamic_resistance, rarc_gpu, 
    water_storage, max_water_storage, soil_moisture_old, soil_moisture_critical, wilting_point, 
    root_gpu, rmin_gpu, LAI_gpu, cv_gpu, f_n
)# Eq. (5), 🚧 TODO: check the gsm_inv multiplication


            println("Sample transpiration[4,22,1,1]: ", Array(transpiration[4:4, 22:22, 1:1, 1:1])[1])
            println("Sample soil_moisture_old[4,22,1]: ", Array(soil_moisture_old[4:4, 22:22, 1:1])[1])


            @timeit to "calculate_soil_evaporation" soil_evaporation = calculate_soil_evaporation(soil_moisture_old, soil_moisture_max, potential_evaporation, b_infilt_gpu, cv_gpu)
            
            # === Update Water Storage with Throughfall Computation; Eq. 16 ===
            @timeit to "update_water_canopy_storage" (water_storage, throughfall) = update_water_canopy_storage(
                water_storage, prec_gpu, cv_gpu, canopy_evaporation, max_water_storage, throughfall, coverage_gpu
            )

            # Set throughfall for bare soil tile (full precip, no canopy interception/evap)
            throughfall[:, :, :, end:end] = prec_gpu .* cv_gpu[:, :, :, end:end]
            
            # Now throughfall has values for all tiles (veg + bare); proceed to surface_runoff
            surface_runoff = calculate_surface_runoff(prec_gpu, throughfall, soil_moisture_old, soil_moisture_max, b_infilt_gpu, cv_gpu)

            # Direct surface runoff
#            @timeit to "calculate_surface_runoff" surface_runoff = calculate_surface_runoff(
#                prec_gpu, throughfall, soil_moisture_old, soil_moisture_max, b_infilt_gpu, cv_gpu
#            ) # Eq. (18a) and (18b)

total_input = sum_with_nan_handling(throughfall, 4)
infiltration_raw = total_input .- surface_runoff
infiltration = max.(infiltration_raw, zero(eltype(infiltration_raw)))

# Optional debugging:
 println("neg. infiltration cells = ", count(<(0), Array(infiltration_raw)))

            soil_moisture_new, subsurface_runoff, Q12 = solve_runoff_and_drainage(
                infiltration,         # Water reaching the soil surface (2D array)
#                sum_with_nan_handling(throughfall, 4),         # Water reaching the soil surface (2D array)
                soil_evaporation,       # Evaporation from each layer (3D array)
                transpiration,
                soil_moisture_old,      # Moisture from previous step (3D array)
            #    Wfc_gpu,                # Field capacity (3D array)
                soil_moisture_max,      # Max moisture (porosity) (3D array)
                ksat_gpu,               # Saturated hydraulic conductivity (3D array)
                residual_moisture,      # Residual moisture (3D array)
                expt_gpu,               # Brooks-Corey exponent (3D array)
                # ARNO baseflow parameters for the bottom layer
                Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu # These are 2D arrays
            ) 


            
            soil_moisture_old = soil_moisture_new


            # Section 2.5
            @timeit to "compute_total_fluxes" begin
                total_et = calculate_total_evapotranspiration(canopy_evaporation, transpiration, soil_evaporation, cv_gpu)
                total_runoff = calculate_total_runoff(surface_runoff, subsurface_runoff, cv_gpu) # TODO: should we use subsurface_runoff_total or subsurface_runoff?
            end          

            @timeit to "Set ice_frac sum"        ice_frac         = 0.0
            
            @timeit to "soil_conductivity"       kappa_array      = soil_conductivity( soil_moisture_new, ice_frac, soil_dens_min,
                                                                   bulk_dens_min, quartz_gpu, organic_frac, porosity)
            @timeit to "volumetric_heat_capacity" cs_array        = volumetric_heat_capacity(
                                                                   bulk_dens_gpu ./ soil_dens_gpu,
                                                                   soil_moisture_new ./ rho_w, ice_frac, organic_frac)
            @timeit to "estimate_layer_temperature" soil_temperature = estimate_layer_temperature(
                                                                   depth_gpu, dp_gpu, tsurf, soil_temperature, Tavg_gpu)
            
            if day == 1 && year == start_year 
                @timeit to "solve_surface_temperature" tsurf = solve_surface_temperature(
                    tsurf,          # Surface temperature (CuArray)
                    soil_temperature,    # Soil layer 1 temperature (CuArray) 
                    albedo_gpu,        # Surface albedo (CuArray)
                    swdown_gpu,   # Rs: shortwave radiation (float_type)
                    lwdown_gpu,   # RL: longwave radiation (float_type)
                    sum_with_nan_handling(cv_gpu .* aerodynamic_resistance, 4), # Aerodynamic resistance 
                    kappa_array,         # Thermal conductivity 
                    depth_gpu,             
                    day_sec,               # TODO: should be 1 instead of day_sec? (time step in seconds -> 1 day in s)
                    cs_array,
                    total_et,
                    tair_gpu,
                    cv_gpu
                ) 

                @timeit to "compute_aerodynamic_resistance" aerodynamic_resistance = compute_aerodynamic_resistance(
                    z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu
                )  # Eq. (3). 

            end

# before calling solve_surface_temperature
ra_eff_inv = sum(cv_gpu ./ aerodynamic_resistance, dims=4)
ra_eff = 1.0 ./ max.(ra_eff_inv, eps(eltype(ra_eff_inv)))

            @timeit to "solve_surface_temperature" tsurf = solve_surface_temperature(
                tsurf,          # Surface temperature (CuArray)
                soil_temperature,    # Soil layer 1 temperature (CuArray) 
                albedo_gpu,        # Surface albedo (CuArray)
                swdown_gpu,   # Rs: shortwave radiation (float_type)
                lwdown_gpu,   # RL: longwave radiation (float_type)
                ra_eff,
                kappa_array,         # Thermal conductivity (CuArray)
                depth_gpu,             
                day_sec,               # TODO: should be 1 instead of day_sec? (time step in seconds -> 1 day in s)
                cs_array,
                total_et,
                tair_gpu,
                cv_gpu
            ) 

println("3 BEFORE OUTPUT Soil moisture at position [4,22,1]: ", Array(soil_moisture_new[4:4, 22:22, 1:1])[1])


@timeit to "outputs" begin
    # ---- GPU-safe sanitizers that keep each array's eltype ----
    san_nan = A -> begin
        T = eltype(A); thr = T(fillvalue_threshold); rep = T(NaN)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    san_zero = A -> begin
        T = eltype(A); thr = T(fillvalue_threshold); rep = T(0.0)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    convcv = A -> convert.(eltype(A), cv_gpu)  # cv cast to A's eltype

    # --- direct outputs ---
    tsurf_output[:, :, day] = Array(tsurf)

    aerodynamic_resistance_output[:, :, day, :]     = Array(aerodynamic_resistance)
    aerodynamic_resistance_summed_output[:, :, day] = Array(sum_with_nan_handling(aerodynamic_resistance, 4))

    transpiration_output[:, :, day, :]         = Array(transpiration)
    transpiration_summed_output[:, :, day]     = Array(sum_with_nan_handling(transpiration, 4))

    tair_output[:, :, day]                     = Array(tair_gpu)
    precipitation_output[:, :, day]            = Array(prec_gpu)
    throughfall_output[:, :, day, :]           = Array(throughfall)
    throughfall_summed_output[:, :, day]       = Array(sum_with_nan_handling(throughfall, 4))

    # --- processed fields (GPU-safe ifelse with array-typed literals) ---
    Q12_processed = san_zero(Q12)
    Q12_output[:, :, day, :] = Array(Q12_processed)

    soil_evaporation_output[:, :, day, :] = Array(soil_evaporation)
    soil_temperature_output[:, :, day, :] = Array(soil_temperature)
    soil_moisture_output[:, :, day, :]    = Array(soil_moisture_new)

    total_et_output[:, :, day]     = Array(total_et)
    total_runoff_output[:, :, day] = Array(total_runoff)
    kappa_array_output[:, :, day,:]= Array(kappa_array)
    cs_array_output[:, :, day,:]   = Array(cs_array)

    potential_evaporation_processed = san_nan(potential_evaporation)
    potential_evaporation_output[:, :, day, :]     = Array(potential_evaporation_processed)
    potential_evaporation_summed_output[:, :, day] = Array(
        sum_with_nan_handling(convcv(potential_evaporation_processed) .* potential_evaporation_processed, 4)
    )

# water storage (grid-cell)
water_storage_processed = san_nan(water_storage)               # still per-canopy
water_storage_output[:, :, day, :]     = Array(water_storage_processed)
water_storage_summed_output[:, :, day] = Array(
    sum_with_nan_handling(convcv(water_storage_processed) .* water_storage_processed, 4)
)



    net_radiation_processed = san_nan(net_radiation)
    net_radiation_output[:, :, day, :]      = Array(net_radiation_processed)
    net_radiation_summed_output[:, :, day]  = Array(
        sum_with_nan_handling(convcv(net_radiation_processed) .* net_radiation_processed, 4)
    )

# --- canopy evaporation (make 4-D output a grid-cell field) ---
canopy_evaporation_processed = san_nan(canopy_evaporation)
canopy_evaporation_gc = convcv(canopy_evaporation_processed) .* canopy_evaporation_processed
canopy_evaporation_output[:, :, day, :]      = Array(canopy_evaporation_gc)
canopy_evaporation_summed_output[:, :, day]  = Array(
    sum_with_nan_handling(canopy_evaporation_gc, 4)
)

# max water storage (grid-cell)
max_water_storage_processed = san_nan(max_water_storage)       # still per-canopy
max_water_storage_output[:, :, day, :]     = Array(max_water_storage_processed)
max_water_storage_summed_output[:, :, day] = Array(
    sum_with_nan_handling(convcv(max_water_storage_processed) .* max_water_storage_processed, 4)
)

    wilting_point_output[:, :, :]          = Array(wilting_point)
    soil_moisture_critical_output[:, :, :] = Array(soil_moisture_critical)
    soil_moisture_max_output[:, :, :]      = Array(soil_moisture_max)
    E_1_t_output[:, :, day, :]             = Array(E_1_t)
    E_2_t_output[:, :, day, :]             = Array(E_2_t)
    residual_moisture_output[:, :, day, :] = Array(residual_moisture)
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
            Base.GC.gc()  # Force garbage collection after processing each year TODO: necessary?
            CUDA.reclaim() # TODO: Check: does this actually do anything?
        end
    else
        println("Skipping year $year due to missing input files.")
    end
end
println("Done!")