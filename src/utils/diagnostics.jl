# Diagnostic utilities for debugging and analyzing simulation outputs.

function run_spike_diagnostics(day, transpiration, soil_moisture_old, 
                               soil_moisture_critical, wilting_point, root_gpu, 
                               cv_gpu, water_storage, max_water_storage, f_n, 
                               potential_evaporation)
    println("\n=== SPIKE FINDER DIAGNOSTIC - Day $day ===")
    
    transp_array = Array(transpiration)
    max_transp = maximum(transp_array)
    high_transp_indices = findall(x -> x > 4.0, transp_array)
    
    println("Maximum transpiration: $max_transp mm/day")
    println("Number of cells with transpiration > 4.0 mm/day: $(length(high_transp_indices))")
    
    if length(high_transp_indices) > 0
        for (idx_num, cart_idx) in enumerate(high_transp_indices[1:min(3, length(high_transp_indices))])
            i, j, k, veg = cart_idx.I
            print_spike_location_info(
                idx_num, i, j, k, veg, transp_array, soil_moisture_old,
                soil_moisture_critical, wilting_point, root_gpu, cv_gpu,
                water_storage, max_water_storage, f_n, potential_evaporation
            )
        end
    else
        println("No transpiration spikes found on this day.")
    end
    
    println("=== END SPIKE FINDER ===\n")
end

function print_spike_location_info(idx_num, i, j, k, veg, transp_array, 
                                   soil_moisture_old, soil_moisture_critical, 
                                   wilting_point, root_gpu, cv_gpu, 
                                   water_storage, max_water_storage, f_n, 
                                   potential_evaporation)
    transp_val = transp_array[i, j, k, veg]
    
    println("\n--- Spike Location #$idx_num: Grid ($i,$j) Veg_tile=$veg ---")
    println("Transpiration: $transp_val mm/day")
    
    # Soil moisture and stress factors
    W1 = Array(soil_moisture_old[i:i, j:j, 1:1])[1]
    W2 = Array(soil_moisture_old[i:i, j:j, 2:2])[1]
    Wcr1 = Array(soil_moisture_critical[i:i, j:j, 1:1])[1]
    Wcr2 = Array(soil_moisture_critical[i:i, j:j, 2:2])[1]
    Wwp1 = Array(wilting_point[i:i, j:j, 1:1])[1]
    Wwp2 = Array(wilting_point[i:i, j:j, 2:2])[1]
    
    g1 = max(0.0, min(1.0, (W1 - Wwp1) / (Wcr1 - Wwp1 + 1e-9)))
    g2 = max(0.0, min(1.0, (W2 - Wwp2) / (Wcr2 - Wwp2 + 1e-9)))
    
    f1 = Array(root_gpu[i:i, j:j, 1:1, veg:veg])[1]
    f2 = Array(root_gpu[i:i, j:j, 2:2, veg:veg])[1]
    g_sw = (f1 * g1 + f2 * g2) / (f1 + f2 + 1e-9)
    
    # Canopy factors
    cv = Array(cv_gpu[i:i, j:j, 1:1, veg:veg])[1]
    ws = Array(water_storage[i:i, j:j, 1:1, veg:veg])[1]
    mws = Array(max_water_storage[i:i, j:j, 1:1, veg:veg])[1]
    fn = Array(f_n[i:i, j:j, 1:1, veg:veg])[1]
    
    W_can = ws / max(cv, 1e-9)
    Wratio = max(0.0, min(1.0, W_can / max(mws, 1e-9)))
    wetFrac = Wratio^(2.0/3.0)
    dry_time_factor = max(0.0, min(1.0, 1.0 - fn * wetFrac))
    
    PE = Array(potential_evaporation[i:i, j:j, 1:1, veg:veg])[1]
    
    # Print diagnostics
    println("Soil moisture: L1=$W1 mm, L2=$W2 mm")
    println("Soil stress: g1=$g1, g2=$g2, combined=$g_sw")
    println("Root fractions: f1=$f1, f2=$f2")
    println("Coverage: cv=$cv")
    println("Canopy: storage=$ws, max=$mws")
    println("Factors: f_n=$fn, dry_factor=$dry_time_factor")
    println("Potential evap: PE=$PE mm/day")
    println("Components: $PE × $g_sw × $dry_time_factor = $(PE * g_sw * dry_time_factor)")
    
    # Identify culprit
    if PE > 10.0
        println("*** CULPRIT: VERY HIGH POTENTIAL EVAPORATION ***")
    elseif g_sw > 0.9 && PE > 5.0
        println("*** CULPRIT: HIGH PE + LOW SOIL STRESS ***")
    elseif dry_time_factor > 0.95 && PE > 3.0
        println("*** CULPRIT: HIGH PE + DRY CANOPY ***")
    end
end

function run_external_debug(day, g_sw_1, g_sw_2, root_gpu, transpiration)
    println("\n=== EXTERNAL DEBUG - Day $day ===")
    
    i, j, veg = 7, 19, 1
    
    g1_val = Array(g_sw_1)[i, j]
    g2_val = Array(g_sw_2)[i, j]
    f1_val = Array(root_gpu)[i, j, 1, veg]
    f2_val = Array(root_gpu)[i, j, 2, veg]
    transp_val = Array(transpiration)[i, j, 1, veg]
    
    layer2_should_dominate = (g2_val >= 0.99) && (f2_val >= 0.5)
    layer1_should_dominate = (g1_val >= 0.99) && (f1_val >= 0.5) && !layer2_should_dominate
    
    expected_gsw = if layer2_should_dominate
        1.0
    elseif layer1_should_dominate
        1.0
    else
        (f1_val * g1_val + f2_val * g2_val) / (f1_val + f2_val)
    end
    
    println("Soil stress: g1=$g1_val, g2=$g2_val")
    println("Root fractions: f1=$f1_val, f2=$f2_val")
    println("Layer 2 should dominate: $layer2_should_dominate")
    println("Layer 1 should dominate: $layer1_should_dominate")
    println("Expected g_sw: $expected_gsw")
    println("Final transpiration: $transp_val mm/day")
    
    println("=== END EXTERNAL DEBUG ===\n")
end