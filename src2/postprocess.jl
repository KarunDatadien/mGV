@kernel function fused_preprocess_kernel!(
    pe_out, nr_out, tr_out, ce_out, ws_out, # 2D Outputs
    @Const(pe_in), @Const(nr_in),       # 4D Inputs
    @Const(tr_in), @Const(ce_in), @Const(ws_in), # 4D Inputs
    @Const(coverage), @Const(cv),       # 4D Weights
    threshold, fill_val                 # Scalars
)
    i, j = @index(Global, NTuple)

    # 1. Initialize Accumulators (sums for this grid cell)
    # We use the type of the output array to ensure stability
    acc_pe = zero(eltype(pe_out))
    acc_nr = zero(eltype(nr_out))
    acc_tr = zero(eltype(tr_out))
    acc_ce = zero(eltype(ce_out))
    acc_ws = zero(eltype(ws_out))

    # 2. Iterate over Vegetation Tiles (4th Dimension)
    # We assume inputs are (nx, ny, 1, n_tiles)
    n_tiles = size(pe_in, 4)
    total_cv = zero(eltype(pe_out))

    # Pre-read bare soil PE (last tile = bare soil, k=n_tiles)
    # Used for VIC's within-tile fcanopy blending:
    # VIC: pe_tile = fcanopy * pe_veg + (1-fcanopy) * pe_soil
    _pe_soil_raw = pe_in[i, j, 1, n_tiles]
    pe_soil = isnan(_pe_soil_raw) || abs(_pe_soil_raw) > threshold ? zero(eltype(pe_out)) : eltype(pe_out)(_pe_soil_raw)

    for k in 1:n_tiles
        # A. Shared Weights
        _cv_raw = cv[i, j, 1, k]
        w_cv = ifelse(isnan(_cv_raw),  zero(eltype(pe_out)), eltype(pe_out)(_cv_raw))
        
        _cov_raw = coverage[i, j, 1, k]
        w_cov = ifelse(isnan(_cov_raw), zero(eltype(pe_out)), eltype(pe_out)(_cov_raw))
        
        total_cv += w_cv

        # B. Potential Evaporation (PE) - simple Cv-weighted sum matching VIC's OUT_PET.
        val = pe_in[i, j, 1, k]
        acc_pe += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(pe_out)), w_cv * val)

        # C. Net Radiation (NR)
        val = nr_in[i, j, 1, k]
        acc_nr += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(nr_out)), w_cv * val)

        # D. Transpiration (TR)
        val = tr_in[i, j, 1, k]
        acc_tr += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(tr_out)), w_cov * val)

        # E. Canopy Evaporation (CE)
        val = ce_in[i, j, 1, k]
        acc_ce += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(ce_out)), w_cv * w_cov * val)

        # F. Water Storage (WS)
        val = ws_in[i, j, 1, k]
        acc_ws += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(ws_out)), w_cv * w_cov * val)
    end

    # 3. Write Final Sums or NaN to Global Memory
    active = !isnan(total_cv) & (total_cv >= eltype(pe_out)(1f-6))
    pe_out[i, j] = ifelse(active, acc_pe, fill_val)
    nr_out[i, j] = ifelse(active, acc_nr, fill_val)
    tr_out[i, j] = ifelse(active, acc_tr, fill_val)
    ce_out[i, j] = ifelse(active, acc_ce, fill_val)
    ws_out[i, j] = ifelse(active, acc_ws, fill_val)
end

@kwdef struct Results{M, T}
    surface_temperature::M # tsurf
    air_temperature::M # tair
    precipitation::M # prec
    total_evapotranspiration::M # total_et
    surface_runoff::M # surface_runoff
    total_runoff::M
    discharge::M # discharge
    travel_time::M # travel_time
    potential_evaporation::M # pe_summed
    net_radiation::M # nr_summed
    transpiration::M # tr_summed
    canopy_evaporation::M # ce_summed
    water_storage::M # ws_summed
    snow_water_equivalent::M # swe_summed
    snow_albedo::M
    snow_surface_temperature::M
    snow_coverage::M
    snow_melt::M
    soil_evaporation::M
    soil_moisture::T  # 3D (nx, ny, nlayers)
end

@adapt_structure Results

"""
4D Snow Aggregation: sum_{b,v}(state[b,v] * Cv[v] * AreaFract[b])
"""
function agg_4d_snow(swe, coverage, melt, snow_fraction, veg_fraction)
    af4 = reshape(
        snow_fraction,
        size(snow_fraction, 1),
        size(snow_fraction, 2),
        size(snow_fraction, 3),
        1
    )
    w4 = (
        ifelse.(
            isnan.(af4),
            0f0,
            af4
        ) .*
        ifelse.(
            isnan.(veg_fraction),
            0f0,
            veg_fraction
        )
    )

    swe_summed = dropdims(sum(ifelse.(isnan.(swe), 0f0, swe) .* w4, dims=(3,4)), dims=(3,4))
    sc_summed = dropdims(sum(ifelse.(isnan.(coverage), 0f0, coverage) .* w4, dims=(3,4)), dims=(3,4))
    sm_summed = dropdims(sum(ifelse.(isnan.(melt), 0f0, melt) .* w4, dims=(3,4)), dims=(3,4))
    return swe_summed, sc_summed, sm_summed
end


function process_daily_outputs(model)
    current_month = month(model.clock.time)

    (;
        surface_temperature, total_evapotranspiration, potential_evaporation, net_radiation
    ) = model.surface_energy_variables
    (; air_temperature, precipitation) = model.forcing_variables
    (; surface_runoff, total_runoff) = model.soil_variables
    (; transpiration, canopy_evaporation, water_storage) = model.canopy_variables
    (; snow_band_area_fraction) = model.grid_parameters

    (; snow_water_equivalent, surface_water, snowpack_water, melt) = model.snow_variables
    snow_albedo = model.snow_variables.albedo
    snow_surface_temperature = model.snow_variables.surface_temperature
    snow_coverage = model.snow_variables.coverage

    soil_moisture = model.soil_variables.moisture
    soil_evaporation = model.soil_variables.evaporation

    vegetation_fraction = model.vegetation_parameters.vegetation_fraction # Cv
    canopy_coverage = @view(model.vegetation_parameters.canopy_coverage[:,:,[current_month],:]) # coverage

    (; fillvalue_threshold) = model.config

    # 1. Allocate Output Arrays (2D)
    nx, ny = (length(model.grid_parameters.longitude), length(model.grid_parameters.latitude))

    pe_summed = similar(surface_temperature, nx, ny)
    nr_summed = similar(surface_temperature, nx, ny)
    tr_summed = similar(surface_temperature, nx, ny)
    ce_summed = similar(surface_temperature, nx, ny)
    ws_summed = similar(surface_temperature, nx, ny)

    # 2. Launch the Fused Kernel
    kernel_launcher! = fused_preprocess_kernel!(device_backend)
    kernel_launcher!(
        pe_summed, nr_summed, tr_summed, ce_summed, ws_summed,
        potential_evaporation, net_radiation, transpiration, canopy_evaporation, water_storage,
        canopy_coverage, vegetation_fraction,
        fillvalue_threshold, NaN32;
        ndrange=(nx, ny)
    )

    # 3. Handle reshapes (metadata only, instant)
    if model.config.enable_routing
        discharge_2d = reshape(model.routing.discharge, size(total_runoff))
        travel_time_2d = reshape(model.routing.travel_time, size(total_runoff)) 
    else
        discharge_2d = similar(total_runoff)
        travel_time_2d = similar(total_runoff)
        fill!(discharge_2d, 0f0)
        fill!(travel_time_2d, 0f0)
    end

    # 4D Snow Aggregation
    swe_summed, sc_summed, sm_summed = agg_4d_snow(
        snow_water_equivalent, snow_coverage, melt, snow_band_area_fraction, vegetation_fraction
    )

    # Land mask: cells with no active bands/veg → NaN (ocean cells)
    af_sum = dropdims(sum(ifelse.(isnan.(vegetation_fraction), 0f0, vegetation_fraction), dims=(3,4)), dims=(3,4))
    land_mask = af_sum .> 1f-6

    swe_masked      = ifelse.(land_mask, swe_summed, NaN32)
    coverage_masked = ifelse.(land_mask, sc_summed, NaN32)
    melt_masked     = ifelse.(land_mask, sm_summed, NaN32)

    # Snow-presence mask: cells with meaningful snow coverage
    snow_present = swe_masked .> 0f0

    # Albedo/surf_temp: weighted by Cv*AreaFract ONLY where snow is present, normalized by that same sum!    
    _w4_base = let
        af4 = reshape(snow_band_area_fraction, size(snow_band_area_fraction,1), size(snow_band_area_fraction,2), size(snow_band_area_fraction,3), 1)
        ifelse.(isnan.(af4), 0f0, af4) .* ifelse.(isnan.(vegetation_fraction), 0f0, vegetation_fraction)
    end
    
    _w4_active_snow = ifelse.(snow_water_equivalent .> 0f0, _w4_base, 0f0)
    cv_snow_sum = max.(dropdims(sum(_w4_active_snow, dims=(3,4)), dims=(3,4)), 1f-6)

    # Compute 2D snow_albedo and snow_surf_temp before tsurf blend
    snow_albedo_2d = let
        num = dropdims(
            sum(ifelse.(isnan.(snow_albedo), 0f0, snow_albedo) .* _w4_active_snow, dims=(3,4));
            dims=(3,4)
        )
        ifelse.(snow_present, num ./ cv_snow_sum, NaN32)
    end

    snow_surf_temp_2d = let
        num = dropdims(sum(ifelse.(isnan.(snow_surface_temperature), 0f0, snow_surface_temperature) .* _w4_active_snow, dims=(3,4)), dims=(3,4))
        ifelse.(snow_present, num ./ cv_snow_sum, NaN32)
    end

    # -----------------------------------------------------------------------
    # Blend tsurf with snow surface temperature (matches VIC's OUT_SURF_TEMP)
    # VIC: energy.Tsurf = snow.surf_temp when snow is present → the reported  
    # surface temperature is cold (near 0°C) over snow, not the bare-soil temp.
    # Our tsurf is solved from the vegetation/soil energy balance; it stays
    # warm even when the cell is snow-covered. We correct this by blending:
    #   tsurf_out = snow_cov * snow_surf_temp + (1 - snow_cov) * bare_tsurf. TODO: is this reasonable?
    # -----------------------------------------------------------------------
    snow_cov_safe = ifelse.(snow_present, coverage_masked, 0f0)
    snow_t_safe   = ifelse.(snow_present, snow_surf_temp_2d, surface_temperature)
    tsurf_blended = snow_cov_safe .* snow_t_safe .+ (1f0 .- snow_cov_safe) .* surface_temperature

    return Results(
        surface_temperature=tsurf_blended,
        air_temperature=air_temperature,
        precipitation=precipitation,
        total_evapotranspiration=total_evapotranspiration,
        surface_runoff=surface_runoff,
        total_runoff=total_runoff,
        discharge=discharge_2d,
        travel_time=travel_time_2d,
        potential_evaporation=pe_summed,
        net_radiation=nr_summed,
        transpiration=tr_summed,
        canopy_evaporation=ce_summed,
        water_storage=ws_summed,
        snow_water_equivalent=swe_masked,
        snow_albedo=snow_albedo_2d,
        snow_surface_temperature=snow_surf_temp_2d,
        snow_coverage=coverage_masked,
        snow_melt=melt_masked,
        soil_evaporation=soil_evaporation,
        soil_moisture=soil_moisture,
    )
end
