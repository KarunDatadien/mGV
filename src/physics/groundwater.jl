function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)
    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions
    Kdry_min = (0.135 .* bulk_dens_min .+ 64.7) ./ (soil_dens_min .- 0.947 .* bulk_dens_min)
    Kdry = (1 .- organic_frac) .* Kdry_min .+ organic_frac .* Kdry_org

    # Fractional degree of saturation
    Sr = ifelse.(porosity .> 0, moist ./ porosity, 0.0)

    # Compute Ks of mineral soil based on quartz content
    Ks_min = ifelse.((quartz .< 0.2) .& (quartz .<= 1.0),
                    7.7 .^ quartz .* 3.0 .^ (1.0 .- quartz),
                    ifelse.(quartz .<= 1.0,
                            7.7 .^ quartz .* 2.2 .^ (1.0 .- quartz),
                            0.0))

    Ks = (1 .- organic_frac) .* Ks_min .+ organic_frac .* Ks_org

    # Calculate Ksat depending on whether the soil is unfrozen (Wu == moist) or partially frozen
    Ksat = ifelse.(Wu .== moist,
                  Ks .^ (1.0 .- porosity) .* Kw .^ porosity,
                  Ks .^ (1.0 .- porosity) .* Ki .^ (porosity .- Wu) .* Kw .^ Wu)

    # Compute the effective saturation parameter, Ke
    Ke = ifelse.(Wu .== moist,
                0.7 .* log10.(max.(Sr, 1e-10)) .+ 1.0,
                Sr)

    # Final Kappa calculation using ifelse to handle moist > 0 condition
    Kappa = ifelse.(moist .> 0,
                   max.((Ksat .- Kdry) .* Ke .+ Kdry, Kdry),
                   Kdry)

    return Kappa
end

function volumetric_heat_capacity(soil_fract, water_fract, ice_fract, organic_frac)
    # Constant values are volumetric heat capacities in J/m^3/K
    Cs = 2.0e6 .* soil_fract .* (1 .- organic_frac) .+
         2.7e6 .* soil_fract .* organic_frac .+
         4.2e6 .* water_fract .+
         1.9e6 .* ice_fract .+
         1.3e3 .* (1.0 .- (soil_fract .+ water_fract .+ ice_fract))  # Air component

    return Cs
end

function calculate_gsm_inv(soil_moisture, soil_moisture_critical, wilting_point)
    ## Initialize gsm_inv to zeros (handles full stress case: soil_moisture < wilting_point)
    # println("soil_moisture shape: ", size(soil_moisture))

    gsm_inv = CUDA.zeros(eltype(soil_moisture), size(soil_moisture,1), size(soil_moisture,2), size(soil_moisture,3) )
    # println("gsm_inv shape: ", size(gsm_inv))
    
    # Calculate the partial stress term for all elements
    partial_stress = (soil_moisture .- wilting_point) ./ (soil_moisture_critical .- wilting_point)

    # Use ifelse to handle the two remaining cases:
    # - Case 1: No stress (soil_moisture >= soil_moisture_critical) -> 1
    # - Case 2: Partial stress (wilting_point <= soil_moisture < soil_moisture_critical) -> partial_stress
    # - Case 3: Anything still zero is implicitly soil_moisture < wilting_point

    gsm_inv .= ifelse.(soil_moisture .>= soil_moisture_critical,
                      1.0,
                      partial_stress)

    return gsm_inv
end


"""
    calculate_interlayer_drainage(Ksat, current_moisture, max_moisture, Wfc, expt)

Calculates the gravitational drainage (flux) between two soil layers based on the
Brooks-Corey formulation used in the VIC model.
"""
function calculate_interlayer_drainage(Ksat, current_moisture, max_moisture, residual_moisture, expt)
    drainage_ratio = (current_moisture .- residual_moisture) ./ (max_moisture .- residual_moisture)
    drainage_ratio = clamp.(drainage_ratio, 0.0f0, 1.0f0)
    Q12 = Ksat .* drainage_ratio .^ expt
    return clamp.(Q12, 0.0f0, Inf32)
end

#function calculate_interlayer_drainage(
#    Ksat,
#    current_moisture,
#    residual_moisture,
#    max_moisture,
#    expt
#)
#    # Epsilon for numerical stability
#    EPSILON = float_type(1e-9)
#
#    # Effective moisture above residual
#    eff_moist = max.(current_moisture .- residual_moisture, 0.0f0)
#    max_eff_moist = max.(max_moisture .- residual_moisture, EPSILON)
#
#    # Term 1: (initial effective moisture)^{1 - expt}
#    term1 = (eff_moist .+ EPSILON) .^ (1.0f0 .- expt)
#
#    # Drainage potential over the time step: Ksat / (max_eff_moist ^ expt) * (1 - expt)
#drainage_potential = (Ksat ./ (max_eff_moist .^ expt)) .* (expt .- 1.0f0)
#
#    # Inner power base: term1 - drainage_potential
#    inner_pow_base = term1 .- drainage_potential
#
#    # Clamp inner_pow_base to prevent negative base in power
#    inner_pow_base = max.(inner_pow_base, 0.0f0)
#
#    # New effective moisture after drainage: (inner_pow_base)^{1 / (1 - expt)}
#    # Handle expt ==1 special case (linear drainage)
#    inv_denom = 1.0f0 ./ (1.0f0 .- expt .+ EPSILON)
#    final_eff_moist = inner_pow_base .^ inv_denom
#
#    # Total drainage: initial eff_moist - final_eff_moist
#    Q12 = eff_moist .- final_eff_moist
#
#    # Ensure drainage is non-negative and doesn't exceed available eff_moist
#    Q12 = clamp.(Q12, 0.0f0, eff_moist)
#
#    return clamp.(Q12, 0.0f0, Inf32)
#end


"""
    calculate_baseflow(bottom_moisture, resid_moist, max_moist, Dsmax, Ds, Ws, c_expt)

Calculates baseflow from the bottom soil layer using the ARNO model.
"""
function calculate_baseflow(bottom_moisture, resid_moist, max_moist, Dsmax, Ds, Ws, c_expt)
    EPSILON = 1.0f-9
    rel_moist = (bottom_moisture .- resid_moist) ./ (max_moist .- resid_moist .+ EPSILON)
    rel_moist = clamp.(rel_moist, 0.0f0, 1.0f0)
    frac = Dsmax .* Ds ./ Ws
    baseflow = frac .* rel_moist
    is_above_ws = rel_moist .> Ws
    frac_nonlinear = (rel_moist .- Ws) ./ (1.0f0 .- Ws .+ EPSILON)
    baseflow .+= is_above_ws .* Dsmax .* (1.0f0 .- Ds ./ Ws) .* (frac_nonlinear .^ c_expt)
    return clamp.(baseflow, 0.0f0, Inf32)
end


"""
Solves the runoff and drainage for a multi-layered soil column.

This function calculates the movement of water between soil layers (drainage),
the runoff from the surface, and the baseflow from the bottom layer. It then
updates the soil moisture for each layer based on a water balance that now
correctly includes inflows (surface water, drainage from above) and outflows
(bare soil evaporation, plant transpiration, drainage to below, baseflow).

Args:
    surface_inflow: Water reaching the soil surface (2D array).
    soil_evaporation: Evaporation from bare soil in each layer (3D array).
                      NOTE: This is assumed to only have non-zero values in the first layer.
    transpiration: Water uptake by plants from each layer (3D array).
                   NOTE: This array dictates which layers lose water to transpiration.
    soil_moisture_old: Moisture from the previous time step (3D array).
    Wfc_gpu: Field capacity of each layer (3D array).
    soil_moisture_max: Maximum moisture (porosity) of each layer (3D array).
    ksat_gpu: Saturated hydraulic conductivity of each layer (3D array).
    residual_moisture: Residual moisture of each layer (3D array).
    expt_gpu: Brooks-Corey exponent for each layer (3D array).
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu: ARNO baseflow parameters (2D arrays).

Returns:
    A tuple containing:
    - soil_moisture_new: The updated soil moisture for each layer (3D array).
    - baseflow: The calculated baseflow from the bottom layer (2D array).
    - Q12: The drainage flux between layers (3D array).
"""

function solve_runoff_and_drainage(
    surface_inflow,      # (ny,nx)  [mm]
    soil_evaporation,    # (ny,nx,nlayer,nveg)  [mm]
    transpiration,       # (ny,nx,nlayer,nveg)  [mm]
    soil_moisture_old,   # (ny,nx,nlayer)       [mm]
    soil_moisture_max,   # (ny,nx,nlayer)       [mm]
    ksat_gpu,            # (ny,nx,nlayer)
    residual_moisture,   # (ny,nx,nlayer)
    expt_gpu,            # (ny,nx,nlayer)
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu       # (ny,nx)
)
    # ---- unify dtype on GPU ----
    T  = eltype(soil_moisture_old)
    Z  = T(0); O = T(1); EPS = T(1e-9)

    W_old  = T.(soil_moisture_old)
    Wmax   = T.(soil_moisture_max)
    Wres   = T.(residual_moisture)
    Ksat   = T.(ksat_gpu)
    exptT  = T.(expt_gpu)
    DsmaxT = T.(Dsmax_gpu); DsT = T.(Ds_gpu)
    WsT    = T.(Ws_gpu);    cexpT = T.(c_expt_gpu)

    inflow = T.(surface_inflow)                     # (ny,nx)
    L      = size(W_old, 3)

    # aggregate E and T per layer (sum over veg)
    EvL = CUDA.zeros(T, size(W_old))                # only layer 1 is used
    TrL = CUDA.zeros(T, size(W_old))
    nTL = min(L, size(transpiration,3))
    TrL[:, :, 1:nTL] .= sum(T.(transpiration), dims=4)[:, :, 1:nTL, 1]
    EvL[:, :, 1]     .= sum(T.(soil_evaporation), dims=4)[:, :, 1, 1]

    # working state
    W = copy(W_old)
    Q12 = CUDA.zeros(T, size(W,1), size(W,2), max(L-1,0))

    # -------- LAYER 1 --------
    # add infiltration
    W[:, :, 1] .+= inflow

    # evap + transpiration (cap by available above residual)
    avail1 = max.(W[:, :, 1] .- Wres[:, :, 1], Z)
    loss1_pot = EvL[:, :, 1] .+ TrL[:, :, 1]
    scale1 = min.(O, avail1 ./ max.(loss1_pot, EPS))
    Ev1 = EvL[:, :, 1] .* scale1
    Tr1 = TrL[:, :, 1] .* scale1
    W[:, :, 1] .-= (Ev1 .+ Tr1)

    # gravitational drainage from updated state
    if L >= 2
        q12_1 = calculate_interlayer_drainage(
                    Ksat[:, :, 1], W[:, :, 1], Wmax[:, :, 1], Wres[:, :, 1], exptT[:, :, 1]
                )
        # cap by available above residual
        avail1 = max.(W[:, :, 1] .- Wres[:, :, 1], Z)
        q12_1 = min.(q12_1, avail1)
        W[:, :, 1] .-= q12_1

        # spill any excess above Wmax to the next layer this step
        spill1 = max.(W[:, :, 1] .- Wmax[:, :, 1], Z)
        W[:, :, 1] .-= spill1
        q12_1 .+= spill1
        Q12[:, :, 1] .= q12_1
    end

    # -------- INTERIOR LAYERS (2..L-1) --------
    for l in 2:max(L-1,1)
        # add inflow from the layer above
        in_l = (l == 2 ? Q12[:, :, 1] : Q12[:, :, l-1])
        W[:, :, l] .+= in_l

        # transpiration (no soil evap here)
        avail = max.(W[:, :, l] .- Wres[:, :, l], Z)
        loss_pot = TrL[:, :, l]
        scale = min.(O, avail ./ max.(loss_pot, EPS))
        Trl = TrL[:, :, l] .* scale
        W[:, :, l] .-= Trl

        # drainage to next layer using UPDATED W
        q12_l = calculate_interlayer_drainage(
                    Ksat[:, :, l], W[:, :, l], Wmax[:, :, l], Wres[:, :, l], exptT[:, :, l]
                )
        avail = max.(W[:, :, l] .- Wres[:, :, l], Z)
        q12_l = min.(q12_l, avail)
        W[:, :, l] .-= q12_l

        # spill over-capacity
        spill = max.(W[:, :, l] .- Wmax[:, :, l], Z)
        W[:, :, l] .-= spill
        q12_l .+= spill

        Q12[:, :, l] .= q12_l
    end

    # -------- BOTTOM LAYER --------
    if L == 1
        Wpre = W[:, :, 1]
        bf_pot = calculate_baseflow(Wpre, Wres[:, :, 1], Wmax[:, :, 1], DsmaxT, DsT, WsT, cexpT)
        avail = max.(Wpre .- Wres[:, :, 1], Z)
        bf    = min.(bf_pot, avail)
        W[:, :, 1] .= Wpre .- bf
        baseflow = bf
    else
        inN = Q12[:, :, L-1]
        W[:, :, L] .+= inN

        bf_pot = calculate_baseflow(W[:, :, L], Wres[:, :, L], Wmax[:, :, L],
                                    DsmaxT, DsT, WsT, cexpT)
        avail  = max.(W[:, :, L] .- Wres[:, :, L], Z)
        bf     = min.(bf_pot, avail)
        W[:, :, L] .-= bf

        # deep percolation if still above capacity
        deep = max.(W[:, :, L] .- Wmax[:, :, L], Z)
        W[:, :, L] .-= deep
        baseflow = bf .+ deep
    end

    # safety clamp (no-op if logic above is right)
    W = min.(max.(W, Wres), Wmax)

    # mass-balance diag (internal Q12 cancels)
    dS = sum(W .- W_old, dims=3)[:, :, 1]
    total_in  = inflow
    total_out = Ev1 .+ sum(TrL, dims=3)[:, :, 1] .+ baseflow
    wb_err = total_in .- total_out .- dS
    println("Max |water balance error| = ", maximum(abs.(wb_err)))

    return W, baseflow, Q12
end

