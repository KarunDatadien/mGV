function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu)
    # Use one element type everywhere
    T   = eltype(cv_gpu)
    Kt  = T(K); gt = T(g); Tf = T(t_freeze); Ric = T(Ri_cr)

    # numeric safeties
    z_floor    = T(1e-3)         # min roughness [m]
    d_floor    = T(1e-2)         # min (z2 - d0) [m]
    wind_floor = T(0.1)          # min wind [m/s]
    L2_min     = T(log(1.01)^2)  # min (ln(arg))^2
    ra_min     = T(1.0)          # clamp bounds [s/m]
    ra_max     = T(1e5)

    # roughness per tile (veg tiles from z0, last tile from soil z0)
    roughness = CUDA.zeros(T, size(cv_gpu))
    roughness[:, :, :, 1:end-1] .= max.(T.(z0_gpu[:, :, :, 1:end-1]), z_floor)
    roughness[:, :, :, end:end] .= max.(T.(z0soil_gpu),               z_floor)

    # effective height above displacement
    z2T   = T(z2)
    d_eff = max.(z2T .- T.(d0_gpu), d_floor)                      # 4D

    # log-law pieces
    ratio = clamp.(d_eff ./ roughness, T(1e-6), T(1e6))           # > 0
    L     = log.(ratio)
    L2    = max.(L .* L, L2_min)
    a_sq  = (Kt^T(2)) ./ L2
    ccoef = T(49.82) .* a_sq .* sqrt.(ratio)

    # stability: bulk Richardson number
    Tmean = max.(((T.(tair_gpu) .+ Tf) .+ (T.(tsurf) .+ Tf)) .* T(0.5), T(100.0))  # 2D
    w     = max.(T.(wind_gpu), wind_floor)                                         # 2D
    Ri_B  = gt .* (T.(tair_gpu) .- T.(tsurf)) .* d_eff ./ (Tmean .* (w .* w))      # 4D via broadcast
    Ri_B  = clamp.(Ri_B, T(-0.5), Ric)

    # friction factor Fw
    Fw_neg = T(1) .- (T(9.4) .* Ri_B) ./ (T(1) .+ ccoef .* sqrt.(abs.(Ri_B)))
    Fw_pos = T(1) ./ (T(1) .+ T(4.7) .* Ri_B).^T(2)
    Fw     = ifelse.(Ri_B .< T(0), Fw_neg, Fw_pos)
    Fw     = clamp.(Fw, T(1e-3), T(10.0))

    # transfer coeff and aerodynamic resistance
    C_H = T(1.351) .* a_sq .* Fw
    C_H = max.(C_H, T(1e-6))
    ra  = T(1.0) ./ (C_H .* w)                                     # broadcast 2D w over 4D C_H
    ra  = clamp.(ra, ra_min, ra_max)

    return ra
end




function compute_partial_canopy_resistance(rmin_gpu, LAI_gpu)
    # Canopy resistance based on soil moisture (Eq. 6), without gsm multiplication; done in evapotranspiration calculation step   
    return rmin_gpu ./ LAI_gpu
end

function calculate_net_radiation(swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
    return (1.0 .- albedo_gpu) .* swdown_gpu .+ emissivity .* (lwdown_gpu .- sigma .* (tsurf .+ 273.15).^4)
end

function calculate_potential_evaporation(
    tair_gpu, vp_gpu, elev_gpu,
    net_radiation, aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
)
    # Element type hygiene
    T = eltype(aerodynamic_resistance)

    # --- Common met terms (2-D, cast to T) ---
    vpd         = max.(T.(calculate_vpd(tair_gpu, vp_gpu)), T(0))
    slope       = T.(calculate_svp_slope(tair_gpu))
    latent_heat = T.(calculate_latent_heat(tair_gpu .+ T(273.15)))
    scale_h     = T.(calculate_scale_height(tair_gpu, elev_gpu))
    p_sfc       = T(p_std) .* exp.(-T.(elev_gpu) ./ scale_h)
    gamma_      = T(1628.6) .* p_sfc ./ latent_heat
    air_dens    = T(0.003486) .* p_sfc ./ (T(273.15) .+ T.(tair_gpu))

    # --- Tile counts & slices (use ra as reference) ---
    N_all   = size(aerodynamic_resistance, 4)        # veg + bare
    @assert N_all ≥ 1 "aerodynamic_resistance must have ≥1 tiles"
    veg_dim = max(N_all - 1, 0)                      # vegetation tiles
    if veg_dim == 0
        # no vegetation tiles: just soil PET
        Rn_soil = T.(net_radiation[:, :, :, end:end])
        ra_soil = aerodynamic_resistance[:, :, :, end:end]
        num_s   = slope .* (Rn_soil .* day_sec) .+ (air_dens .* c_p_air .* vpd .* day_sec ./ ra_soil)
        den_s   = latent_heat .* (slope .+ gamma_ .* (1 .+ T(0.0) ./ ra_soil))  # SOIL_RARC=0
        pe = max.(num_s ./ den_s, T(0))
        return pe
    end

    # Views that GUARANTEE matching tile counts
    @views begin
        Rn_veg   = T.(net_radiation[:, :, :, 1:veg_dim])
        rc_veg   = rmin_gpu[:, :, :, 1:veg_dim] ./ max.(LAI_gpu[:, :, :, 1:veg_dim], T(1e-6))
        rarc_veg = rarc_gpu[:, :, :, 1:veg_dim]
        ra_veg   = aerodynamic_resistance[:, :, :, 1:veg_dim]

        Rn_soil  = T.(net_radiation[:, :, :, N_all:N_all])
        ra_soil  = aerodynamic_resistance[:, :, :, N_all:N_all]
    end

    # --- Vegetation PET (Penman–Monteith with rc=rmin/LAI) ---
    num_v = slope .* (Rn_veg .* day_sec) .+ (air_dens .* c_p_air .* vpd .* day_sec ./ ra_veg)
    den_v = latent_heat .* (slope .+ gamma_ .* (1 .+ (rc_veg .+ rarc_veg) ./ ra_veg))
    pe_veg = max.(num_v ./ den_v, T(0))

    # --- Bare soil PET (PM with rc=0) ---
    SOIL_RARC = T(0.0)
    num_s = slope .* (Rn_soil .* day_sec) .+ (air_dens .* c_p_air .* vpd .* day_sec ./ ra_soil)
    den_s = latent_heat .* (slope .+ gamma_ .* (1 .+ SOIL_RARC ./ ra_soil))
    pe_soil = max.(num_s ./ den_s, T(0))

    # --- Assemble full 4-D output with N_all tiles ---
    pe = CUDA.zeros(T, size(net_radiation))
    pe[:, :, :, 1:veg_dim] .= pe_veg
    pe[:, :, :, N_all:N_all] .= pe_soil
    return pe
end


function calculate_max_water_storage(LAI_gpu, cv_gpu, coverage_gpu)
    # Compute maximum water intercepted/stored in the canopy cover
    result = K_L .* LAI_gpu #TODO should we multiply by .* cv_gpu ?
    return ifelse.(isnan.(result) .| (abs.(result) .> fillvalue_threshold), 0.0, result)
end

        using Statistics


function calculate_canopy_evaporation(
    water_storage, max_water_storage, potential_evaporation,
    aerodynamic_resistance, rarc, prec_gpu, cv_gpu, rmin, LAI_gpu,
    tair_gpu, elev_gpu
)
    # ---- sanitize ----
    potential_evaporation .= ifelse.(isnan.(potential_evaporation) .| (abs.(potential_evaporation) .> fillvalue_threshold), 0.0, potential_evaporation)
    water_storage         .= ifelse.(isnan.(water_storage)         .| (abs.(water_storage)         .> fillvalue_threshold), 0.0, water_storage)
    max_water_storage     .= ifelse.(isnan.(max_water_storage)     .| (abs.(max_water_storage)     .> fillvalue_threshold), 0.0, max_water_storage)
    aerodynamic_resistance.= ifelse.(isnan.(aerodynamic_resistance).| (abs.(aerodynamic_resistance).> fillvalue_threshold), 1e6, aerodynamic_resistance)
    rarc                  .= ifelse.(isnan.(rarc)                  .| (abs.(rarc)                  .> fillvalue_threshold), 0.0, rarc)

    # ---- Δ and γ (same as your PET code) ----
    slope            = calculate_svp_slope(tair_gpu)                 # Pa / °C
    latent_heat      = calculate_latent_heat(tair_gpu .+ 273.15)     # J / kg
    scale_height     = calculate_scale_height(tair_gpu, elev_gpu)    # m
    surface_pressure = p_std .* exp.(-elev_gpu ./ scale_height)      # Pa
    gamma_           = 1628.6 .* surface_pressure ./ latent_heat     # Pa / °C

    # ---- resistances ----
    T       = eltype(potential_evaporation)
    tiny    = T(1e-12)
    rc      = rmin ./ max.(LAI_gpu, T(1e-6))                         # r_c = rmin / LAI
    ra      = aerodynamic_resistance

    # OPTIONAL: temporary floor to test if rα is too small (set to nothing to disable)
    # e.g., RALPHA_MIN = T(80.0)   # try 50–150 s m^-1 if diagnostics show ra_ratio ≈ 1
    RALPHA_MIN = nothing
    ralpha = (RALPHA_MIN === nothing) ? rarc : max.(rarc, RALPHA_MIN)

    # ---- convert your veg PET (with rc) -> wet-canopy PET (rc = 0) via PM denominator ratio ----
    den_rc  = slope .+ gamma_ .* (1 .+ (rc .+ ralpha) ./ ra)         # PET denominator you used
    den_w   = slope .+ gamma_ .* (1 .+ ralpha ./ ra)                 # rc = 0, keep rα
    
    E_p_wet = potential_evaporation .* (den_rc ./ max.(den_w, tiny)) # mm / Δt

    # ---- VIC Eq. (1): (W/Wm)^(2/3) * E_p_wet * ra/(ra + rα) ----
    Wratio = clamp.(water_storage ./ max.(max_water_storage, tiny), 0.0, 1.0)
    ra_ratio = ra ./ max.(ra .+ ralpha, tiny)
    canopy_evaporation_star = (Wratio .^ (2/3)) .* E_p_wet .* ra_ratio

    # ---- Rain-limited fraction f_n (Eq. 10) — **no cv here** ----
    f_n = clamp.((water_storage .+ prec_gpu) ./ max.(canopy_evaporation_star, tiny), 0.0, 1.0)

    # ---- Actual canopy evaporation (per-canopy units) ----
    canopy_evaporation = f_n .* canopy_evaporation_star
    canopy_evaporation = ifelse.(isnan.(canopy_evaporation) .| (abs.(canopy_evaporation) .> fillvalue_threshold), 0.0, canopy_evaporation)

    # Bare-soil tile has no canopy
    canopy_evaporation[:, :, :, end:end] .= 0.0

    # ---- lightweight diagnostics (sampled) ----
    if rand() < 1.0
        @info "canopy diag" med_ra=Statistics.median(Array(ra))
        @info "canopy diag" med_ralpha=Statistics.median(Array(ralpha))
        @info "canopy diag" med_ra_ratio=Statistics.median(Array(ra_ratio))
        @info "canopy diag" med_W23=Statistics.median(Array((Wratio .^ (2/3))))
        @info "canopy diag" med_Epwet=Statistics.median(Array(E_p_wet))
    end

    return canopy_evaporation, f_n
end




#=
This function has been updated to fix the GPU type-compatibility error.

The problem was caused by mixing input arrays of different precisions
(Float64 and Float32). The fix is to explicitly convert the Float64
inputs to Float32 at the beginning of the function. This ensures all
subsequent calculations are type-stable and GPU-compatible.
=#

using CUDA
using Printf

function calculate_transpiration(
    potential_evaporation::CuArray, aerodynamic_resistance::CuArray, rarc_gpu::CuArray,
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray,
    soil_moisture_critical::CuArray, wilting_point::CuArray, root_gpu::CuArray,
    rmin_gpu::CuArray, LAI_gpu::CuArray, cv_gpu, f_n
)
    # -------- One dtype everywhere --------
    T   = eltype(potential_evaporation)
    F0  = T(0); F1 = T(1); EPS = T(1e-9)

    # Cast only the arrays used below that may be Float64
    Wcr_T  = T.(soil_moisture_critical)      # (ny,nx,layer)
    Wwp_T  = T.(wilting_point)               # (ny,nx,layer)
    Wmax_T = T.(max_water_storage)           # (ny,nx,1,nveg)
    W_T    = T.(water_storage)               # (ny,nx,1,nveg)
    cv_T   = T.(cv_gpu)                      # (ny,nx,1,nveg)
    PE_T   = T.(potential_evaporation)       # (ny,nx,1,nveg)

    # -------- soil-moisture stress per layer (assume L=2) --------
    W1   = @view soil_moisture_old[:, :, 1]
    W2   = @view soil_moisture_old[:, :, 2]
    Wcr1 = @view Wcr_T[:, :, 1]
    Wcr2 = @view Wcr_T[:, :, 2]
    Wwp1 = @view Wwp_T[:, :, 1]
    Wwp2 = @view Wwp_T[:, :, 2]

    g1 = clamp.((W1 .- Wwp1) ./ (Wcr1 .- Wwp1 .+ EPS), F0, F1)
    g2 = clamp.((W2 .- Wwp2) ./ (Wcr2 .- Wwp2 .+ EPS), F0, F1)

    ny, nx = size(g1)
    veg_dim = size(root_gpu, 4)   # vegetation tiles (exclude bare)
    nveg    = size(cv_T, 4)       # vegetation + bare

    f1  = reshape(@view(root_gpu[:, :, 1, :]), ny, nx, 1, veg_dim)
    f2  = reshape(@view(root_gpu[:, :, 2, :]), ny, nx, 1, veg_dim)
    g1b = reshape(g1, ny, nx, 1, 1)
    g2b = reshape(g2, ny, nx, 1, 1)

    sumf     = sum(root_gpu, dims=3)                          # (ny,nx,1,veg_dim)
    g_sw_veg = clamp.((f1 .* g1b .+ f2 .* g2b) ./ (sumf .+ EPS), F0, F1)

    # -------- canopy wetness (per canopy) --------
    cv_safe = max.(cv_T, EPS)
    W_can   = W_T ./ cv_safe                                   # canopy-area basis
    Wratio  = clamp.(W_can ./ max.(Wmax_T, EPS), F0, F1)
    wetFrac = Wratio .^ (T(2)/T(3))
    dry_time_factor = clamp.(F1 .- T.(f_n) .* wetFrac, F0, F1)
    dry_time_factor[:, :, :, end:end] .= F1                             # bare soil unaffected

    # -------- transpiration for vegetation tiles only --------
transpiration_veg =
    cv_T[:, :, :, 1:veg_dim] .*
    dry_time_factor[:, :, :, 1:veg_dim] .*
    PE_T[:, :, :, 1:veg_dim] .*
    g_sw_veg

    transpiration_veg = clamp.(transpiration_veg, F0, T(Inf))

    # split to layers (weights)
    denom_veg = f1 .* g1b .+ f2 .* g2b .+ EPS
    E_1_t_veg = transpiration_veg .* (f1 .* g1b) ./ denom_veg
    E_2_t_veg = transpiration_veg .* (f2 .* g2b) ./ denom_veg

    # GPU-safe NaN guards
    E_1_t_veg = ifelse.(isnan.(E_1_t_veg), F0, E_1_t_veg)
    E_2_t_veg = ifelse.(isnan.(E_2_t_veg), F0, E_2_t_veg)

    # -------- assemble full-tile arrays with bare slice = 0 --------
    transpiration_full = CUDA.zeros(T, ny, nx, 1, nveg)
    E_1_t_full         = CUDA.zeros(T, ny, nx, 1, nveg)
    E_2_t_full         = CUDA.zeros(T, ny, nx, 1, nveg)

    transpiration_full[:, :, :, 1:veg_dim] .= transpiration_veg
    E_1_t_full[:, :, :, 1:veg_dim]        .= E_1_t_veg
    E_2_t_full[:, :, :, 1:veg_dim]        .= E_2_t_veg

    # diagnostic g_sw_total (2D)
    f1t = sum(@view(root_gpu[:, :, 1, :]), dims=3)[:, :, 1]
    f2t = sum(@view(root_gpu[:, :, 2, :]), dims=3)[:, :, 1]
    g_sw_total = clamp.((f1t .* g1 .+ f2t .* g2) ./ (f1t .+ f2t .+ EPS), F0, F1)

    return transpiration_full, E_1_t_full, E_2_t_full, g1, g2, g_sw_total
end






#TODO: investigate issue and cleanup function below
# Issue: SOIL EVAP WARNING: Found 402 cells where topsoil_moisture_max is zero. This will cause division by zero.

# You might need to import the Printf module to use @printf
using Printf
function calculate_soil_evaporation(soil_moisture, soil_moisture_max, potential_evaporation, b_i, cv_gpu)
    # Use the grid's shared top-layer moisture (NOT the bare-soil tile slice)
    # Shapes:
    #   soil_moisture, soil_moisture_max :: (ny, nx, nlayer)
    #   potential_evaporation            :: (ny, nx, 1, nveg)
    #   cv_gpu                           :: (ny, nx, 1, nveg)

    T = eltype(potential_evaporation)
    EPS = T(1e-9)

    # 2D fields (ny, nx)
    topsoil_moisture     = @view soil_moisture[:, :, 1]
    topsoil_moisture_max = @view soil_moisture_max[:, :, 1]

    # Guard rails
    moisture_ratio = clamp.(topsoil_moisture ./ (topsoil_moisture_max .+ EPS), T(0), T(1))

    # A_sat (Xinanjiang curve)
    A_sat = T(1) .- (T(1) .- moisture_ratio) .^ b_i
    A_sat = clamp.(A_sat, T(0), T(1))
    x = T(1) .- A_sat

    # Series S(b) from Liang 1994 (Eq. 15) — 2D
    S_series = CUDA.ones(T, size(x))
    @inbounds for n = 1:20
        S_series .+= (b_i ./ (n .+ b_i)) .* (x .^ (n ./ b_i))
    end

    # Infiltration ratio (Eq. 13) — 2D
    infiltration_ratio = T(1) .- (T(1) .- A_sat) .^ (T(1) ./ b_i)

    # Broadcast 2D fields to the bare-soil PET slice (ny, nx, 1, 1)
    A_sat4   = reshape(A_sat,   size(A_sat,1),   size(A_sat,2),   1, 1)
    x4       = reshape(x,       size(x,1),       size(x,2),       1, 1)
    S4       = reshape(S_series,size(S_series,1),size(S_series,2),1, 1)
    infil4   = reshape(infiltration_ratio, size(infiltration_ratio,1), size(infiltration_ratio,2), 1, 1)

    # Bare-soil potential evaporation slice
    Ep_soil = potential_evaporation[:, :, :, end:end]  # (ny, nx, 1, 1)

    # Unsaturated + saturated contributions (Eqs. 14–15)
    Ev_unsat = Ep_soil .* infil4 .* x4 .* S4
    Ev_sat   = Ep_soil .* A_sat4

    total_soil_evaporation = Ev_sat .+ Ev_unsat

    # Scale by bare-soil fraction (apply cv only here)
    scaled_soil_evaporation = total_soil_evaporation .* cv_gpu[:, :, :, end:end]

    return scaled_soil_evaporation
end




#function update_water_canopy_storage(water_storage, prec_gpu, cv_gpu, canopy_evaporation, max_water_storage, throughfall, coverage)
#
#    # Calculate new water storage: current storage + (precipitation - canopy evaporation)
##    new_water_storage = water_storage .+ (prec_gpu .* cv_gpu .* coverage) .- canopy_evaporation
#    new_water_storage = water_storage .+ (prec_gpu .* cv_gpu) .- canopy_evaporation
#
#    # Compute throughfall: excess water beyond max storage
#    throughfall = max.(0, new_water_storage .- max_water_storage)
#    
#    # Update water storage: clamp between 0 and max_water_storage
#    water_storage = max.(0.0, min.(new_water_storage, max_water_storage))
#    
#    return (water_storage), throughfall 
#end

function update_water_canopy_storage(water_storage, prec, cv, canopy_evap, Wm, throughfall, coverage)
    # Canopy water balance is per canopy area (Liang 1994, Eq. 16)
    new_storage = water_storage .+ prec .- canopy_evap
    excess      = max.(0.0, new_storage .- Wm)           # P′_l in Eq. 16 (only when W hits Wm)
    water_storage = clamp.(new_storage, 0.0, Wm)

    # Throughfall that reaches the ground must be grid-area weighted
    throughfall = cv .* excess
    return water_storage, throughfall
end




# Eq. (23): Total evapotranspiration
function calculate_total_evapotranspiration(canopy_evaporation, transpiration, soil_evaporation, cv_gpu)
    # Sum canopy evaporation and transpiration for vegetated classes (n = 1:nveg-1)
    vegetated_et = cv_gpu[:, :, :, 1:end-1] .* (canopy_evaporation[:, :, :, 1:end-1]) .+ transpiration[:, :, :, 1:end-1] # cv_gpu[:, :, :, 1:end-1] .* 
    
    # Add bare soil evaporation (n = nveg)
    bare_soil_et =  soil_evaporation # cv_gpu[:, :, :, end:end] .*
    
    # Total evapotranspiration (sum across cover classes)
    total_et = sum_with_nan_handling(vegetated_et, 4) .+ bare_soil_et
  
    return total_et
end