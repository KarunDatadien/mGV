function solve_surface_temperature(
    tsurf::CuArray, T1::CuArray, T2::CuArray, albedo::CuArray,
    Rs::CuArray, RL::CuArray, rh::CuArray, kappa::CuArray,
    D1, D2, delta_t, Cs::CuArray, E_n::CuArray
)
    
    # Debugging prints (synchronize GPU before printing)
#    println("Cs has NaN: ", any(isnan, Cs), " min/max: ", minimum(Cs), " / ", maximum(Cs))
#    println("kappa has NaN: ", any(isnan, kappa), " min/max: ", minimum(kappa), " / ", maximum(kappa))
#    println("rh has NaN: ", any(isnan, rh), " min/max: ", minimum(rh), " / ", maximum(rh))

    # === Compute dependent quantities ===
    latent_heat = calculate_latent_heat(tsurf) # should be with dimension W*m^-2
    rho_a = 1.225 # Density of air (TODO: make temperature dependent?)

#    println("latent_heat min/max: ", minimum(latent_heat), " / ", maximum(latent_heat))

    # === Constants ===
    top_layer_depth = 0.3 # TODO: good value?
    rho_w = 1000.0  # Density of liquid water (TODO: make temperature dependent?)

    # === Precompute constants ===
#    base_term = (kappa ./ D2) .+ (Cs .* D2 ./ (2 * delta_t))
#
#    println("Shape of kappa: ", size(kappa))
#    println("Shape of Cs: ", size(Cs))
#    println("Shape of D1: ", size(D1))
#    println("Shape of D2: ", size(D2))
#    println("Shape of base_term: ", size(base_term))
#
#    heat_transfer_term = base_term ./ (1 .+ (D1 / D2) .+ (Cs .* D1 .* D2 ./ (2 * delta_t .* kappa)))
#    air_term = (rho_a .* c_p_air ./ rh) .+ (rho_a .* c_p_air .* top_layer_depth ./ (2 * delta_t))
#    common_term = heat_transfer_term .+ air_term

    kappa_top = kappa[:, :, 1]  # Select top layer (2D: nx × ny)
    Cs_top = Cs[:, :, 1]        # Select top layer (2D: nx × ny)
    base_term = (kappa_top ./ D2) .+ (Cs_top .* D2 ./ (2 * delta_t))

   # println("Shape of kappa_top: ", size(kappa_top))
   # println("Shape of Cs_top: ", size(Cs_top))
   # println("Shape of D1: ", size(D1))
   # println("Shape of D2: ", size(D2))
   # println("Shape of base_term: ", size(base_term))

    heat_transfer_term = base_term ./ (1 .+ (D1 ./ D2) .+ (Cs_top .* D1 .* D2 ./ (2 * delta_t .* kappa_top)))
    air_term = (rho_a .* c_p_air ./ rh[:, :, 1, 1]) .+ (rho_a .* c_p_air .* top_layer_depth ./ (2 * delta_t))  # Adjust rh to 2D
    common_term = heat_transfer_term .+ air_term

    # Extend common_term to 4D for nveg
    common_term_4d = repeat(common_term, outer=(1, 1, 1, nveg))  # (204, 180, 1, 22)

#
    #println("base_term min/max: ", minimum(base_term), " / ", maximum(base_term))
    #println("heat_transfer_term min/max: ", minimum(heat_transfer_term), " / ", maximum(heat_transfer_term))
    #println("air_term min/max: ", minimum(air_term), " / ", maximum(air_term))
    #println("common_term min/max: ", minimum(common_term), " / ", maximum(common_term))

    # === Define the residual function f(Ts_new) = lhs - rhs ===
    function f(Ts_new, Ts_old)
        # Debug shapes
 #       println("Shape of Ts_new: ", size(Ts_new))
     #   println("Shape of Ts_old: ", size(Ts_old))
     #   println("Shape of albedo: ", size(albedo))
     #   println("Shape of rh: ", size(rh))
     #   println("Shape of E_n: ", size(E_n))

        lhs = emissivity .* sigma .* Ts_new.^4 .+ common_term_4d .* Ts_new
      #  println("Shape of lhs: ", size(lhs))

        rhs = (1 .- albedo) .* repeat(Rs, outer=(1, 1, 1, nveg)) .+ 
              emissivity .* repeat(RL, outer=(1, 1, 1, nveg)) .+
              (rho_a .* c_p_air ./ rh) .* tsurf .- 
              rho_w .* lat_vap .* (E_n ./ (day_sec .* mm_in_m)) .+
              (rho_a .* c_p_air .* top_layer_depth .* Ts_old ./ (2 * delta_t)) .+
              ((repeat(kappa_top, outer=(1, 1, 1, nveg)) .* repeat(T2, outer=(1, 1, 1, nveg)) ./ repeat(D2, outer=(1, 1, 1, nveg))) .+ 
               (repeat(Cs_top, outer=(1, 1, 1, nveg)) .* repeat(D2, outer=(1, 1, 1, nveg)) .* repeat(T1, outer=(1, 1, 1, nveg)) ./ (2 * delta_t))) ./ 
              (1 .+ (repeat(D1, outer=(1, 1, 1, nveg)) ./ repeat(D2, outer=(1, 1, 1, nveg))) .+ 
               (repeat(Cs_top, outer=(1, 1, 1, nveg)) .* repeat(D1, outer=(1, 1, 1, nveg)) .* repeat(D2, outer=(1, 1, 1, nveg)) ./ 
                (2 * delta_t .* repeat(kappa_top, outer=(1, 1, 1, nveg)))))


        #rhs = (1 .- albedo) .* Rs .+ emissivity .* RL .+
        #      (rho_a .* c_p_air ./ rh) .* tsurf .- rho_w .* lat_vap .* (E_n ./ (day_sec .* mm_in_m)) .+ # E_n: convert mm/day to m/s
        #      (rho_a .* c_p_air .* top_layer_depth .* Ts_old ./ (2 * delta_t)) .+
        #      ((kappa_top .* T2 ./ D2) .+ (Cs_top .* D2 .* T1 ./ (2 * delta_t))) ./ 
        #      (1 .+ (D1 / D2) .+ (Cs_top .* D1 .* D2 ./ (2 * delta_t .* kappa_top)))

    #    println("Shape of rhs: ", size(rhs))

        return lhs .- rhs
    end

    # === Derivative of f(Ts_new) for Newton-Raphson ===
    function df_dTs_new(Ts_new)
        # Derivative w.r.t. Ts_new: d/dTs_new (emissivity * sigma * Ts_new^4 + common_term * Ts_new)
        # = 4 * emissivity * sigma * Ts_new^3 + common_term
        return 4.0 .* emissivity .* sigma .* Ts_new.^3 .+ common_term
    end

    # === Newton-Raphson solver ===
    Ts_old = tsurf  # Initial guess on GPU
    Ts_new = tsurf
#    tolerance = 1e-6
#    max_iter = 10

#    for iter in 1:max_iter
#        residual = f(Ts_new, Ts_old)
#        derivative = df_dTs_new(Ts_new)
#        
#        # Newton step: Ts_new = Ts_new - f(Ts_new) / f'(Ts_new)
#        delta_Ts = residual ./ derivative
#        Ts_new = Ts_new .- delta_Ts
#        println("Shape of delta_Ts: ", size(delta_Ts))
#
#        # Check convergence
#        max_delta = maximum(abs.(delta_Ts))
#        min_delta = minimum(abs.(delta_Ts))
#
#        println("Iteration $iter: Max delta Ts = $max_delta")
#        println("Iteration $iter: Min delta Ts = $min_delta")
#        if max_delta < tolerance
#            println("Converged after $iter iterations")
#            break
#        end
#        
#        # Update Ts_old for next iteration
#        Ts_old = Ts_new
#    end

    Ts_new = f(Ts_new, Ts_old) # TODO: THIS FUNCTION RETURNS 0's, first fix this, then check the iteration loop


    return Ts_new
end

function estimate_layer_temperature(depth_gpu, dp_gpu, tsurf, soil_temperature, Tavg_gpu)
    # Based on Liang et al. (1999): Modeling ground heat flux in land surface 
    # parameterization schemes

    # Assign inputs
    topsoil_temperature = sum(soil_temperature[:, :, 1:2], dims=3) ./ 2  # Average of layers 1-2
    # Taking Tavg_gpu as deep soil temperature

    # Model layer 1 (my layers 1-2): Average of Tsurf and topsoil_temperature
    soil_temperature[:, :, 1:1] .= 0.5 .* (tsurf .+ topsoil_temperature)
    soil_temperature[:, :, 2:2] .= 0.5 .* (tsurf .+ topsoil_temperature)

    # Model layer 2 (my layer 3)
    soil_temperature[:, :, 3:3] .= Tavg_gpu .- (dp_gpu ./ depth_gpu[:, :, 3:3]) .* 
                                   (topsoil_temperature .- Tavg_gpu) .* 
                                   (exp.(-(depth_gpu[:, :, 2:2] .+ depth_gpu[:, :, 3:3]) ./ dp_gpu) .- 
                                    exp.(-depth_gpu[:, :, 2:2] ./ dp_gpu))

    return soil_temperature
end