function solve_surface_temperature(tsurf, soil_temperature, albedo, Rs, RL, rh, kappa, depth_gpu, delta_t, Cs, total_et, T_a, cv_gpu)

    albedo = sum_with_nan_handling(cv_gpu .* albedo, 4)
    albedo .= ifelse.(isnan.(albedo) .| (abs.(albedo) .> 1e30), 0.0, albedo)

    # Ensure inputs are valid (replace NaN or extreme values)
    rh .= ifelse.(isnan.(rh) .| (abs.(rh) .> 1e30), 0.0, rh)
    kappa .= ifelse.(isnan.(kappa) .| (abs.(kappa) .> 1e30), 0.0, kappa)
    Cs .= ifelse.(isnan.(Cs) .| (abs.(Cs) .> 1e30), 0.0, Cs)

    T1_K = soil_temperature[:, :, 2:2] .+ 273.15 # TODO: should be average of layer 1 and 2?
    T2_K = soil_temperature[:, :, 3:3] .+ 273.15

    T1_K .= ifelse.(isnan.(T1_K) .| (abs.(T1_K) .> 1e15), 0.0, T1_K)
    T2_K .= ifelse.(isnan.(T2_K) .| (abs.(T2_K) .> 1e15), 0.0, T2_K)

    D1 = sum(depth_gpu[:, :, 1:2], dims=3) 
    D2 = depth_gpu[:,:, 3:3]

    T_a_K = T_a .+ 273.15
    T_a_K .= ifelse.(isnan.(T_a_K) .| (abs.(T_a_K) .> 1e15), 0.0, T_a_K)

    # === Compute dependent quantities ===
    latent_heat = calculate_latent_heat(tsurf) # should be with dimension W*m^-2

    # === Precompute constants ===
    kappa_top = kappa[:, :, 1]  # Select top layer (2D: nx × ny)
    Cs_top = Cs[:, :, 1]        # Select top layer (2D: nx × ny)

    base_term = (kappa_top ./ D2) .+ (Cs_top .* D2 ./ (2 * delta_t))

    heat_transfer_term = base_term ./ (1 .+ (D1 ./ D2) .+ (Cs_top .* D1 .* D2 ./ (2.0 * delta_t .* kappa_top)))
    air_term = (rho_a .* c_p_air ./ max.(rh, 1e-3)) .+ (rho_a .* c_p_air .* D1 ./ (2.0 * delta_t)) # TODO: dividing by rh here is an issue (?) --> becomes infinite where rh = 0
    common_term = heat_transfer_term .+ air_term


    # === Define the residual function f(Ts_new) = lhs - rhs ===
    function f(Ts_new, Ts_old)

        # Convert to Kelvin for Stefan-Boltzmann term
        Ts_new_K = Ts_new .+ 273.15
        Ts_old_K = Ts_old .+ 273.15

        lhs = emissivity .* sigma .* Ts_new_K.^4 .+ common_term .* Ts_new_K

        # Incoming radiative and turbulent fluxes
        term1 = (1 .- albedo) .* Rs
        term2 = emissivity .* RL
        
        # Sensible‐heat exchange with atmosphere
        term3 = (rho_a .* c_p_air ./ max.(rh, 1e-3)) .* T_a_K  # T_a_K in Kelvin

        # Latent‐heat loss via evapotranspiration
        term4 = rho_w .* calculate_latent_heat(Ts_new_K) .* (total_et ./ (day_sec .* mm_in_m))
        
        # Heat storage change from previous time step
        term5 = (rho_a .* c_p_air .* D1 .* Ts_old_K) ./ (2 * delta_t)
        
        # Conductive exchange with soil layers
        num   = (kappa_top .* T2_K ./ D2) .+
                (Cs_top  .* D2 .* T1_K ./ (2 * delta_t))
        den   = 1 .+
                (D1 ./ D2) .+
                (Cs_top .* D1 .* D2 ./ (2 * delta_t .* kappa_top))
        term6 = num ./ den
        
        # Sum everything up
        rhs = term1 .+ term2 .+ term3 .- term4 .+ term5 .+ term6

        return lhs .- rhs # TODO: fix rhs (and lhs?), rhs gives infinite values (maybe because of division by rh?)
    end

    # === Derivative of f(Ts_new) for Newton-Raphson ===
    function df_dTs_new(Ts_new)
        # Watson correlation parameters
        Hvap_Tb = 2.26e6
        Tb = 373.15
        Tc = 647.096
        n = 0.38
        denom = Tc - Tb
        Ts_new_K = Ts_new .+ 273.15

        # Derivative of latent heat using ifelse
        ratio = (Tc .- Ts_new_K) ./ denom
        ratio = clamp.(ratio, 1e-6, 1.0)
        L_v_deriv = ifelse.(Ts_new_K .< Tc,
                            Hvap_Tb .* n .* (ratio .^ (n - 1)) .* (-1 ./ denom),
                            0.0)

        et_flux = total_et ./ (day_sec .* mm_in_m)
        term4_deriv = rho_w .* L_v_deriv .* et_flux
        
        Ts_new_K = Ts_new .+ 273.15
        return 4.0 .* emissivity .* sigma .* Ts_new_K.^3 .+ common_term .- term4_deriv
    end

    # === Newton-Raphson solver ===
    Ts_old = tsurf  # Initial guess on GPU
    Ts_new = tsurf

    tolerance = 1e-3
    max_iter = 10

    for iter in 1:max_iter
        residual = f(Ts_new, Ts_old)
        derivative = df_dTs_new(Ts_new)
        
        # Newton step with per-grid-point derivative check
        delta_Ts = ifelse.(abs.(derivative) .>= 1e-10, residual ./ derivative, 0.0)
        delta_Ts = clamp.(delta_Ts, -10.0, 10.0)

        # Update convergence status
        converged = abs.(delta_Ts) .< tolerance
        if all(converged)
            println("Converged after $iter iterations")
            break
        end

        # Update Ts_new only for non-converged points
        delta_Ts = ifelse.(converged, 0.0, delta_Ts)
        Ts_new = Ts_new .- delta_Ts
        Ts_new = clamp.(Ts_new, -100.0, 100)  # Prevent unphysical temperatures

        # Log max/min delta for monitoring
        max_delta = maximum(abs.(delta_Ts))  # Use CUDA.maximum for CuArray
        min_delta = minimum(abs.(delta_Ts))  # Use CUDA.minimum for CuArray
        num_converged = sum(converged)  # Number of converged points
        println("Iteration $iter: Number of converged points = $num_converged")
        println("Iteration $iter: Ts_new min/max: ", minimum(Ts_new), " / ", maximum(Ts_new))

        # Update Ts_old for next iteration
        Ts_old = Ts_new
    end

    Ts_new = ifelse.( (Ts_new .== -100.0) .| (Ts_new .== 100.0), 0.0, Ts_new)  # Add this line

    return Ts_new
end

function estimate_layer_temperature(depth_gpu, dp_gpu, tsurf, soil_temperature, Tavg_gpu)
    # Based on Liang et al. (1999): Modeling ground heat flux in land surface parameterization schemes

    # Assign inputs
    topsoil_temperature = sum(soil_temperature[:, :, 1:2], dims=3) ./ 2  # Average of layers 1-2

    # Model layer 1 (my layers 1-2): Average of Tsurf and topsoil_temperature
    soil_temperature[:, :, 1:1] = 0.5 .* (tsurf .+ topsoil_temperature)
    soil_temperature[:, :, 2:2] = 0.5 .* (tsurf .+ topsoil_temperature)

    # Model layer 2 (my layer 3)
    soil_temperature[:, :, 3:3] = Tavg_gpu .- (dp_gpu ./ depth_gpu[:, :, 3:3]) .* 
                                   (topsoil_temperature .- Tavg_gpu) .* 
                                   (exp.(-(depth_gpu[:, :, 2:2] .+ depth_gpu[:, :, 3:3]) ./ dp_gpu) .- 
                                    exp.(-depth_gpu[:, :, 2:2] ./ dp_gpu))

    return soil_temperature
end